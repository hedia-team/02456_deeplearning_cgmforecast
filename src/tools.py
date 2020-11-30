import os
import tempfile

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import code_path
from ray import tune
from torch.autograd import Variable
from tqdm import tqdm

from src.data import DataframeDataLoader
from src.models.hediaNetExample import DilatedNet


# %%
def predict_cgm(data_obj, model: nn.Module) -> np.ndarray:
    """
    Return ANN predictions given a data object and a model
    """

    dset_test = data_obj.load_test_data()

    test_loader = DataframeDataLoader(
        dset_test,
        batch_size=8,
        shuffle=False,
    )

    model.eval()
    all_output = []
    with torch.no_grad():
        for (data, target) in test_loader:
            inputs = data
            inputs = Variable(inputs.permute(0, 2, 1)).contiguous()
            out = model(inputs)

            # If the output is a distribution
            if isinstance(out, torch.distributions.normal.Normal):
                all_output.append(np.hstack([out.mean, out.scale]))

            else:  # Derive the mean or mean & std manually
                # Collect results to either vector or matrix depending on model
                out_all = np.hstack([col.detach().numpy() for i, col in enumerate(out)])
                out_all = out_all.reshape(len(target), -1)

                all_output.append(out_all)

    return np.vstack(all_output)


def train_cgm(config: dict, model_setup=None, data_obj=None, max_epochs=10, n_epochs_stop=5, grace_period=5,
              use_ray_tune=True, checkpoint_dir=None, train_name=None):
    """
    max_epochs : Maximum allowed epochs
    n_epochs_stop : Number of epochs without imporvement in validation error before the training terminates
    grace_period : Number of epochs before termination is allowed
    checkpoint_dir (Optional) : Where to save the model checkpoints

    """
    # Build network
    assert model_setup['type'] in ['gaussian', 'simple'], "Unknown model type!"

    if model_setup['type'] == 'simple':
        model = DilatedNet(h1=config["h1"], h2=config["h2"])
    elif model_setup['type'] == 'gaussian':
        model = DilatedNetGaussian(h1=config["h1"], h2=config["h2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # Optimizer and loss criterion
    # criterion = nn.SmoothL1Loss(reduction='sum')
    criterion = model_setup['loss']
    # criterion = model_setup['loss']
    # criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.RMSprop(
        model.parameters(), lr=config['lr'], weight_decay=config['wd'])  # n

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Load data
    trainset, valset = data_obj.load_train_and_val()

    train_loader = DataframeDataLoader(
        trainset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataframeDataLoader(
        valset,
        batch_size=int(config['batch_size']),
        shuffle=False,
    )

    min_val_loss = np.Inf
    epoch_no_improve = 0

    # Create temporary file for storing checkpoints
    checkpoint_file = tempfile.NamedTemporaryFile(suffix='_checkpoint', delete=False)
    # Add progress bar and hide if multi threading from RayTune is present
    # Consider fixing it, so we can actually see the progress of each thread
    pbar = tqdm(range(max_epochs), position=0, leave=True, desc='Progress', disable=use_ray_tune)

    val_loss_per_batch = 0.0
    n_batches = len(train_loader)

    try:
        for epoch in pbar:  # loop over the dataset multiple times
            epoch_loss = 0.0
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, targets]
                inputs, targets = data
                inputs = Variable(inputs.permute(0, 2, 1)).contiguous()

                if targets.size(0) == int(config['batch_size']):

                    inputs, targets = inputs.to(device), targets.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.reshape(-1, 1))
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    epoch_steps += 1

                    print_every = 20
                    # print every nth mini-batches
                    if i % print_every == (print_every - 1):
                        pbar.set_postfix({'left_of_epoch': f'{round(100 - 100 * i / n_batches, 2):05.2f}%',
                                          'loss_element_in_batch': f'{(running_loss / (print_every * int(config["batch_size"]))):1.2E}',
                                          # Avg loss pr element in batch
                                          'avg_loss_train': f'{(epoch_loss / epoch_steps):1.2E}',
                                          # Avg training loss for epoch
                                          'avg_loss_val': f'{val_loss_per_batch :1.2E}',
                                          'epochs_no_imp': epoch_no_improve
                                          })
                        # print("[%d, %5d] Avg loss pr element in mini batch:  %.3f"
                        #      % (epoch + 1, i + 1,
                        #         running_loss / (print_every*int(config['batch_size']))))

                        running_loss = 0.0  # {(epoch_loss/epoch_steps): 1.2E}

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(val_loader, 0):
                with torch.no_grad():
                    inputs, targets = data
                    inputs = Variable(inputs.permute(0, 2, 1)).contiguous()

                    if targets.size(0) == int(config['batch_size']):
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = model(inputs)

                        loss = criterion(outputs, targets.reshape(-1, 1))
                        val_loss += loss.cpu().numpy()
                        val_steps += 1

            if use_ray_tune:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (model.state_dict(), optimizer.state_dict()), path)

                    tune.report(loss=(val_loss / val_steps))

            # Compute avg validation over number of batches
            val_loss_per_batch = val_loss / val_steps
            if val_loss_per_batch < min_val_loss:
                epoch_no_improve = 0
                min_val_loss = val_loss / val_steps

                if not use_ray_tune:
                    torch.save((model.state_dict(), optimizer.state_dict()), checkpoint_file.name)
                    #print("Saved better model!")

            else:
                epoch_no_improve += 1

            if epoch > grace_period and epoch_no_improve == n_epochs_stop:
                print('Early stopping!')
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Forced early training exit')

    print("Finished Training")

    if not use_ray_tune:
        return checkpoint_file.name


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    '''
    Interpolates between two colors.
    Fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    '''
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def move_single_point(data_frame: pd.DataFrame, feature: str, current_idx: int, offset_min: float):
    """
    Moves a single value of some feature in a dataset back
    offset_min minutes in time
    """

    # Convert from minutes to an index
    offset_idx = int(np.round(offset_min / 5))

    # Make sure we do not go below time 0
    new_idx = np.max((0, current_idx - offset_idx))

    # Move the value
    feature_val = data_frame[feature].iloc[current_idx].copy()
    data_frame[feature].iloc[current_idx] = np.nan  # Set old value to nan
    data_frame[feature].iloc[new_idx] = feature_val  # Insert value into new position


def add_label_noise(data_frame: pd.DataFrame, feature: str, sampleRule):
    """
    Moves all values of specific feature in a dataframe df
    back in time given the rule sampleRule
    """

    # Run through all points with the feature
    feature_idx = np.where(~np.isnan(data_frame[feature]))[0]
    for idx in feature_idx:
        # Choose offset
        offset_min = sampleRule.sample()

        # Move the points
        move_single_point(data_frame=data_frame, feature=feature,
                          current_idx=idx, offset_min=offset_min)

# %%
