# %%
import datetime
import getpass
import json
from shutil import copyfile

import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from config import code_path, data_path, figure_path, model_path
from src.data import DataframeDataLoader
from src.evaluation import evaluateModel
from src.load_data import dataLoader
from src.losses import nll_loss, nll_loss_manual
from src.models.hediaNetExample import DilatedNet
from src.parameter_sets.par import *
from src.tools import train_cgm

#
#
# %load_ext autoreload
# %autoreload 2


# Paths to data, code, figures, etc. should be set in config.py.
# Initialize the config.py file by copying from config.template.py.
# ---------------------------------------------------------------------
# DEFINE MODEL, PARAMETERS AND DATA
# - Change <par> to the name of file containing your parameters
# - Change <hediaNet> to the name of file containing your model architecture and DilatedNet to the name
#   of your model. Also change in train_cgm and optmizeHypers.py
# ---------------------------------------------------------------------


# Tensorboard log setup
# Create a directory for the model if it doesn't already exist
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
user = getpass.getuser()
model_id = f'id_{current_time}_{user}'
model_path_id = model_path / model_id
model_path_id.mkdir(exist_ok=True, parents=True)
model_figure_path = figure_path / model_id
model_figure_path.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------
# DEFINE DATA
# ---------------------------------------------------------------------
# Define data object
data_pars = {}
data_pars['path'] = data_path
data_pars['train_data'] = train_data
data_pars['test_data'] = test_data
data_pars['validation_data'] = test_data

data_pars['start_date_train'] = start_date_train
data_pars['start_date_test'] = start_date_test
data_pars['start_date_validation'] = start_date_val

data_pars['end_date_train'] = end_date_train
data_pars['end_date_test'] = end_date_test
data_pars['end_date_validation'] = end_date_val


data_obj = dataLoader(data_pars, features, n_steps_past=16,
                      n_steps_future=6,
                      allowed_gap=10,
                      scaler=StandardScaler())


# ---------------------------------------------------------------------
# EXTRACT DATA AND TEST THE MODEL
# ---------------------------------------------------------------------
config = {
    "batch_size": 4,
    "lr": 0.00613617,
    "h1": 32,
    "h2": 16,
    "final_1x1_1": 512,
    "final_1x1_2": 64,
    "wd": 0.0201964,
}


model = DilatedNet(h1=config["h1"],
                           h2=config["h2"])


# Load training data
trainset, valset = data_obj.load_train_and_val()

train_loader = DataframeDataLoader(
    trainset,
    batch_size=int(config['batch_size']),
    shuffle=True,
    drop_last=True,
)

# Perform a single prediction
data = next(iter(train_loader))

inputs, targets = data
# It is important to permute the dimensions of the input!!
inputs = Variable(inputs.permute(0, 2, 1)).contiguous()

output = model(inputs)
print(output)
print(targets)


# %%
# ---------------------------------------------------------------------
# TRAING THE MODEL
# ---------------------------------------------------------------------
model_setup = {
    'type': 'simple',
    'loss': torch.nn.SmoothL1Loss(reduction='mean')
}

# Make sure the model archiecture loaded in train_cgm matches the hyper configuration
best_epoch_checkpoint_file_name = train_cgm(config, model_setup, max_epochs=30, grace_period=5,
          n_epochs_stop=15, data_obj=data_obj, use_ray_tune=False)

# Build network
if model_setup['type'] == 'simple':
    model = DilatedNet(h1=config["h1"], h2=config["h2"])
elif model_setup['type'] == 'gaussian':
    model = DilatedNetGaussian(h1=config["h1"], h2=config["h2"])


# Load best model
model_state, optimizer_state = torch.load(best_epoch_checkpoint_file_name)
model.load_state_dict(model_state)

# Copy the trained model to model path
copyfile(best_epoch_checkpoint_file_name,
         model_path_id / 'checkpoint')

#with open(code_path / 'src' / 'model_state_tmp' / 'hyperPars.json', 'w') as fp:
#    json.dump(config, fp)


# %% Evaluate model
# ---------------------------------------------------------------------
# EVALUATE THE MODEL
# ---------------------------------------------------------------------
evaluationConfiguration = {
    'distance': True,
    'hypo': True,
    'clarke': False,
    'parke': True,
    'lag': True,
    'plotLag': True,
    'plotTimeseries': True
}
# ---------------------------------------------------------------------

evalObject = evaluateModel(data_obj, model)

#
if evaluationConfiguration['distance']:
    distance = evalObject.get_distanceAnalysis()
if evaluationConfiguration['hypo']:
    hypo = evalObject.get_hypoAnalysis()
if evaluationConfiguration['lag']:
    lag = evalObject.get_lagAnalysis(figure_path=model_figure_path)
if evaluationConfiguration['plotTimeseries']:
    evalObject.get_timeSeriesPlot(figure_path=model_figure_path)
if evaluationConfiguration['clarke']:
    clarkes, clarkes_prob = evalObject.apply_clarkes_error_grid(
        'mg/dl', figure_path=model_figure_path)
if evaluationConfiguration['parke']:
    parkes, parkes_prob = evalObject.apply_parkes_error_grid(
        'mg/dl', figure_path=model_figure_path)
