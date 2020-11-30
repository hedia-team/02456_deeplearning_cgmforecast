# %%
import datetime
import getpass
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from config import code_path, data_path
from src.data import DataframeDataLoader
from src.load_data import dataLoader
from src.losses import nll_loss, nll_loss_manual
from src.models.hediaNetExample import DilatedNet
from src.tools import train_cgm

pass


# Paths to data, code, figures, etc. should be set in config.py.
# Initialize the config.py file by copying from config.template.py.
# ---------------------------------------------------------------------
# DEFINE MODEL, PARAMETERS AND DATA
# - Change <path_to_hedia_folder> to your path to the Hedia folder
# - Change <par1> to the name of file containing your parameters
# - Change <dilated_cnn_regression> to the name of file containing your model architecture
# ---------------------------------------------------------------------


# %%
def test_rmse(model_to_evaluate, data_obj=None, device="cpu"):
    dset_test = data_obj.load_test_data()

    test_loader = DataframeDataLoader(
        dset_test,
        batch_size=4,
        shuffle=False,
    )

    SSE = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs = Variable(inputs.permute(0, 2, 1)).contiguous()
            inputs, targets = inputs.to(device), targets.to(device)

            out = model_to_evaluate(inputs)

            if isinstance(out, torch.distributions.normal.Normal):
                prediction = out.mean
            else:
                prediction = out[0]

            total += targets.size(0)
            SSE += np.sum(np.power((prediction - targets.reshape(-1, 1)).numpy(), 2))

    return np.sqrt(SSE / total)


def search_best_hypers(model_setup: dict, config_schedule=None, num_samples=10, max_num_epochs=15, n_epochs_stop=2, grace_period=5,
                       gpus_per_trial=0, data_obj=None):
    assert data_obj is not None

    _experiment_id = 'no_name_yet'

    if config_schedule is None:
        config_schedule = {
            "batch_size": tune.choice([4, 8, 16, 32, 64]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "h1": tune.sample_from(lambda: 2 ** np.random.randint(3, 8)),
            "h2": tune.sample_from(lambda: 2 ** np.random.randint(3, 8)),
            "final_1x1_1": tune.sample_from(lambda: 2 ** np.random.randint(6, 11)),
            "final_1x1_2": tune.sample_from(lambda: 2 ** np.random.randint(6, 11)),
            "wd": tune.loguniform(1e-4, 1e-1),
        }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])

    result = tune.run(
        partial(train_cgm, model_setup=model_setup, data_obj=data_obj, n_epochs_stop=n_epochs_stop,
                max_epochs=max_num_epochs, grace_period=grace_period),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config_schedule,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    # Build best network
    # Build network

    # Make sure correct model type
    assert model_setup['type'] in ['gaussian', 'simple'], "Unknown model type!"

    if model_setup['type'] == 'simple':
        best_trained_model = DilatedNet(h1=best_trial.config["h1"], h2=best_trial.config["h2"])

    # Make sure correct model type
    assert model_setup['type'] in ['gaussian', 'simple'], "Unknown model type!"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value

    print("BEST MODEL DIR: ", best_checkpoint_dir)
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # Call load to fit scaler. Should be a better solution
    _, _ = data_obj.load_train_and_val()

    test_rmse_val = test_rmse(best_trained_model, data_obj)
    print("Best trial test set rmse: {}".format(test_rmse_val))

    # Save the results
    experiment = {
        'name': str(_experiment_id),
        'best_trial_dir': str(best_checkpoint_dir),

        'train_data': str(data_obj.train_data),
        'test_data': str(data_obj.test_data),

        'start_date_train': str(data_obj.start_date_train),
        'start_date_test': str(data_obj.start_date_test),

        'end_date_train': str(data_obj.end_date_train),
        'end_date_test': str(data_obj.end_date_test)
    }

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    user = getpass.getuser()
    _experiment_id = f'id_{current_time}_{user}'
    experiment_path = code_path / 'hyper_experiments'  # / model_id
    experiment_path.mkdir(exist_ok=True, parents=True)

    with open(experiment_path / (_experiment_id + '.json'), 'w') as outfile:
        json.dump(experiment, outfile, indent=4)

    ''' Optinally Print information on where optimal model is saved '''
    # print("\n Experiment details are saved in:\n", experiment_path / (_experiment_id + '.json'))
    # print("\n Checkpoint for best configuration is saved in:\n", best_checkpoint_dir)

    return _experiment_id


if __name__ == "__main__":
    # Only load parameters if the program is run as a script
    from src.parameter_sets.par import *

    features = ['CGM', 'CHO', 'insulin', 'CGM_delta']

    model = {
        'type': 'simple',
        'loss': torch.nn.SmoothL1Loss(reduction='mean')  # Choose 'torch.nn.SmoothL1Loss(reduction='mean')' or nll_loss
    }

    # Define data object
    data_pars = {}
    data_pars['path'] = data_path
    data_pars['train_data'] = train_data
    data_pars['test_data'] = test_data
    data_pars['validation_data'] = val_data

    data_pars['start_date_train'] = start_date_train
    data_pars['start_date_test'] = start_date_test
    data_pars['start_date_validation'] = start_date_val

    data_pars['end_date_train'] = end_date_train
    data_pars['end_date_test'] = end_date_test
    data_pars['end_date_validation'] = end_date_val

    data_obj_hyperOpt = dataLoader(data_pars, features, n_steps_past=16,
                                   n_steps_future=6,
                                   allowed_gap=10,
                                   scaler=StandardScaler())

    # Run the searcher for hyper parameters
    experiment_id = search_best_hypers(model, num_samples=10, n_epochs_stop=3, max_num_epochs=5,
                                       gpus_per_trial=0, grace_period=2, data_obj=data_obj_hyperOpt)
