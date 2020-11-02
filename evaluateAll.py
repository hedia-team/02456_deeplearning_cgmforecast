#%%
import datetime
import getpass
import json
import os
from functools import partial
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import code_path, data_path, figure_path, model_path
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from optimizeHypers import searchBestHypers
from src.data import DataframeDataLoader
from src.evaluation import evaluateModel
from src.load_data import dataLoader
from src.models.hediaNetExample import DilatedNet
from src.tools import crosscorr, train_cgm


# Define search parameters
# ----------------------------------------
NUM_SAMPLES = 3 # Number of different hyper parameter setting
MAX_NUM_EPOCHS = 4 # Maximum number of epochs in training
N_EPOCHS_STOP = 2 # Number of consecuetive epochs with no improvement in validation data before terminating training
GRACE_PERIOD = 3 # Minimum number of epochs before termination is allowed
# ----------------------------------------

# Define final train parameters
# ----------------------------------------
MAX_NUM_EPOCHS_FINAL = 4
N_EPOCHS_STOP_FINAL = 2
GRACE_PERIOD_FINAL = 1
# ----------------------------------------



features = ['CGM', 'CHO', 'insulin', 'CGM_delta']

# Define dates
dates = pd.DataFrame(columns = ['start_date_train', 'end_date_train', 'start_date_test', 'end_date_test'])
dates.loc['559-ws-training'] = ['2021-12-07 01:17:00', '2022-01-05 23:56:00', '2022-01-06 00:01:00', '2022-01-17 23:56:00']
dates.loc['563-ws-training'] = ['2021-09-13 12:33:00', '2021-10-13 19:45:00', '2021-10-13 19:50:00', '2021-10-28 23:56:00']
dates.loc['575-ws-training'] = ['2021-11-17 12:04:00', '2021-12-17 21:57:00', '2021-12-17 22:02:00', '2022-01-01 23:55:00']
dates.loc['570-ws-training'] = ['2021-12-07 16:29:00', '2022-01-06 00:11:00', '2022-01-06 00:16:00', '2022-01-16 23:59:00']
dates.loc['591-ws-training'] = ['2021-11-30 17:06:00', '2022-01-02 15:30:00', '2022-01-02 15:35:00', '2022-01-13 23:58:00']
dates.loc['588-ws-training'] = ['2021-08-30 11:53:00', '2021-10-02 14:28:00', '2021-10-02 14:33:00', '2021-10-14 23:55:00']


# Define data set
train_data_sequence = [['575-ws-training'],
                       ['570-ws-training'],
                       ['563-ws-training'],
                       ['559-ws-training'],
                       ['591-ws-training'],
                       ['588-ws-training'],
                       ['575-ws-training','559-ws-training','570-ws-training','559-ws-training','591-ws-training','588-ws-training'],
                       ['575-ws-training','559-ws-training','570-ws-training','559-ws-training','591-ws-training','588-ws-training'],
                       ['575-ws-training','559-ws-training','570-ws-training','559-ws-training','591-ws-training','588-ws-training'],
                       ['575-ws-training','559-ws-training','570-ws-training','559-ws-training','591-ws-training','588-ws-training'],
                       ['575-ws-training','559-ws-training','570-ws-training','559-ws-training','591-ws-training','588-ws-training'],
                       ['575-ws-training','559-ws-training','570-ws-training','559-ws-training','591-ws-training','588-ws-training']
                    ]

test_data_sequence = ['575-ws-training',
                      '570-ws-training',
                      '563-ws-training',
                      '559-ws-training',
                      '591-ws-training',
                      '588-ws-training',
                      '575-ws-training',
                      '570-ws-training',
                      '563-ws-training',
                      '559-ws-training',
                      '591-ws-training',
                      '588-ws-training',
                     ]



scores = pd.DataFrame(columns = ['RMSE', 'MARD', 'MAE', 'A', 'B', 'C', 'D', 'E', 'precision', 'recall', 'F1'])
scores.index.name = '[training], test'

for i in range(12):

    train_data = train_data_sequence[i]
    test_data = test_data_sequence[i]
    start_date_train = list(dates['start_date_train'][train_data])
    end_date_train = list(dates['end_date_train'][train_data])
    start_date_test = dates['start_date_test'][test_data]
    end_date_test = dates['end_date_test'][test_data]

    print("\n \n \n")
    print("--------------------------------------------")
    print("Case #", i)
    print("TRAIN DATA: ", train_data)
    print("TEST DATA: ", test_data)
    print("--------------------------------------------")
    print("\n")


    # Define data object
    data_pars = {}
    data_pars['path'] = data_path
    data_pars['train_data'] = train_data
    data_pars['test_data'] = test_data
    data_pars['validation_data'] = test_data

    data_pars['start_date_train'] = start_date_train
    data_pars['start_date_test'] = start_date_test
    data_pars['start_date_validation'] = start_date_test

    data_pars['end_date_train'] = end_date_train
    data_pars['end_date_test'] = end_date_test
    data_pars['end_date_validation'] = end_date_test



    data_obj_hyperOpt = dataLoader(data_pars, features, n_steps_past=16, 
                                                n_steps_future=6, 
                                                allowed_gap=10, 
                                                scaler=StandardScaler())

    experiment_id = searchBestHypers(num_samples=NUM_SAMPLES, n_epochs_stop=N_EPOCHS_STOP, max_num_epochs=MAX_NUM_EPOCHS, grace_period=GRACE_PERIOD, gpus_per_trial=0, data_obj=data_obj_hyperOpt)
    #experiment_id = main(num_samples=2, n_epochs_stop=3, max_num_epochs=2, gpus_per_trial=0, grace_period=1, data_obj=data_obj_hyperOpt)


    #%%
    print("\n \n")
    print("--------------------------------------------")
    print("Now retrain model with optimal parameters")
    exeriment_path = code_path / 'hyper_experiments' / (experiment_id + '.json')


    with open(exeriment_path) as json_file:
        experiment = json.load(json_file)

    best_model_dir = experiment['best_trial_dir']
    par_file = Path(best_model_dir) / '..' / 'params.json'

    with open(par_file) as json_file: 
        optHyps = json.load(open(par_file)) 


    # Build model
    with open(par_file) as json_file: 
        optHyps = json.load(open(par_file)) 

    model = DilatedNet(h1=optHyps["h1"], 
                        h2=optHyps["h2"])


    data_obj = dataLoader(data_pars, features, n_steps_past=16, 
                                                    n_steps_future=6, 
                                                    allowed_gap=10, 
                                                    scaler=StandardScaler())

    train_cgm(optHyps, max_epochs= MAX_NUM_EPOCHS_FINAL, grace_period=GRACE_PERIOD_FINAL, n_epochs_stop=N_EPOCHS_STOP_FINAL, data_obj=data_obj, useRayTune=False)
    #train_cgm(optHyps, max_epochs= 3, grace_period=1, n_epochs_stop=2, data_obj=data_obj, useRayTune=False)

    # Load best model state
    model_state, optimizer_state = torch.load(os.path.join(
    './src/model_state_tmp', "checkpoint"))
    model.load_state_dict(model_state)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    user = getpass.getuser()
    model_id = f'id_{current_time}_{user}'
    model_id = experiment_id

    model_figure_path = code_path / 'trained_models' / model_id
    model_figure_path.mkdir(exist_ok=True)

    #%%
    # ---------------------------------------------------------------------
    # EVALUATE THE MODEL
    # ---------------------------------------------------------------------
    evaluationConfiguration = {
        'distance': 1,
        'hypo': 1,
        'clarke' : 1,
        'lag': 1,
        'plotLag': 1,
        'plotTimeseries': 1
    }
    # ---------------------------------------------------------------------

    evalObject = evaluateModel(data_obj, model)


    if evaluationConfiguration['distance']: 
        distance = evalObject.get_distanceAnalysis()
    if evaluationConfiguration['hypo']: 
        hypo = evalObject.get_hypoAnalysis()
    if evaluationConfiguration['lag']: 
        lag = evalObject.get_lagAnalysis(figure_path=model_figure_path)
    if evaluationConfiguration['plotTimeseries']: 
        evalObject.get_timeSeriesPlot(figure_path=model_figure_path)
    if evaluationConfiguration['clarke']: 
        clarkes, clarkes_prob = evalObject.clarkesErrorGrid('mg/dl', figure_path=model_figure_path)


    scores.loc[str([train_data, test_data])] = [
        distance['rmse'], distance['mard'], distance['mae'],
        clarkes_prob['A'], clarkes_prob['B'], clarkes_prob['C'], clarkes_prob['D'], clarkes_prob['E'],
        hypo['precision'], hypo['recall'], hypo['F1']
    ]


    scores.to_csv('all_scores.csv')
    data_pars

    copyfile(par_file, model_figure_path / "optPars.json")
    copyfile(code_path / 'hyper_experiments'/ (experiment_id + '.json'), model_figure_path / "data_properties.json")


    
# %%
