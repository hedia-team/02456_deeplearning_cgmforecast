import os
import numpy as np
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from torch.autograd import Variable

from src.data import DataframeDataLoader
from src.models.hediaNetExample import DilatedNet

from typing import Tuple

#%%
def predict_cgm(data_obj, model: nn.Module) -> np.ndarray:

    dset_test = data_obj.load_test_data()

    test_loader = DataframeDataLoader(
        dset_test,
        batch_size=8,
        shuffle=False,
    )

    model.eval()
    outputs = []
    #loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = Variable(data.permute(0, 2, 1)).contiguous()
            target = Variable(target.unsqueeze_(1))
            output = model(data)
            
            outputs.append(output)

    return np.concatenate(outputs).squeeze() 


def train_cgm(config: dict, data_obj=None, max_epochs=10, n_epochs_stop=5, grace_period=5, useRayTune=True, checkpoint_dir=None):
    '''
    max_epochs : Maximum allowed epochs
    n_epochs_stop : Number of epochs without imporvement in validation error before the training terminates
    grace_period : Number of epochs before termination is allowed

    '''
    # Build network
    model = DilatedNet(h1=config["h1"], 
                       h2=config["h2"])

    # Move model between cpu and gpu
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)


    # Optimizser and loss criterion
    #criterion = nn.SmoothL1Loss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=config['wd'])  # n


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
    epochs_no_improve = 0
    early_stop = False

    try:
        for epoch in range(max_epochs):  # loop over the dataset multiple times
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
                    loss = criterion(outputs, targets.reshape(-1,1))
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    epoch_steps += 1

                    print_every = -50
                    if i % print_every == (print_every-1):  # print every nth mini-batches
                        print("[%d, %5d] Avg loss pr element in mini batch: %.3f" % (epoch + 1, i + 1,
                                                        running_loss / (print_every*int(config['batch_size']))))
                        running_loss = 0.0

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

                        loss = criterion(outputs, targets.reshape(-1,1))
                        val_loss += loss.cpu().numpy()
                        val_steps += 1
            
            if useRayTune:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                    tune.report(loss=(val_loss / val_steps)) 
            
            if (val_loss / val_steps) < min_val_loss:
                epoch_no_improve = 0
                min_val_loss = val_loss / val_steps

                if not useRayTune:
                    path = os.path.join('./src/model_state_tmp', "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
                    print("Saved better model!")

            else:
                epoch_no_improve += 1          

            if not useRayTune:
                print('Epoch {0}'.format(epoch+1), end='')
                print(f', Training loss: {(epoch_loss/epoch_steps):1.2E}', end='')
                print(f', Validation loss: {(val_loss/val_steps):1.2E}')
        

            if epoch > grace_period and epoch_no_improve == n_epochs_stop:
                print('Early stopping!' )
                early_stop = True
                break
            
            if early_stop:
                print("Stopped")
                break

        
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Forced early training exit')
        
        
    print("Finished Training")


def my_mean(x):
    return np.average(x, weights=np.ones_like(x) / x.size)


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


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



def move_single_point(df: pd.DataFrame, feature: str, current_idx: int, offset_min: float):
    '''
    Moves a single value of some feature in a dataset back
    offset_min minutes in time
    '''

    # Convert from minutes to an index
    offset_idx = int(np.round(offset_min/5))

    # Make sure we do not go below time 0
    new_idx = np.max((0,current_idx - offset_idx))

    # Move the value
    feature_val = df[feature].iloc[current_idx].copy()
    df[feature].iloc[current_idx] = np.nan # Set old value to nan
    df[feature].iloc[new_idx] = feature_val # Insert value into new position


def addLabelNoise(df: pd.DataFrame, feature: str, sampleRule):
    '''
    Moves all values of specific feature in a dataframe df
    back in time given the rule sampleRule
    '''
    
    # Run through all points with the feature
    feature_idx = np.where(~np.isnan(df[feature]))[0]
    for idx in feature_idx:

        # Choose offset
        offset_min = sampleRule.sample()

        # Move the points
        move_single_point(df=df, feature=feature, current_idx=idx, offset_min=offset_min)



