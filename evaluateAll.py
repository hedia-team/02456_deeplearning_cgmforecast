# %%
import datetime
import getpass
import json
from pathlib import Path
from shutil import copyfile

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import code_path, data_path, figure_path, model_path, result_path
from optimizeHypers import search_best_hypers
from src.evaluation import evaluateModel
from src.load_data import dataLoader
from src.models.hediaNetExample import DilatedNet
from src.parameter_sets.evaluateAllPars import (GRACE_PERIOD, GRACE_PERIOD_FINAL, MAX_NUM_EPOCHS,
                                                MAX_NUM_EPOCHS_FINAL, N_EPOCHS_STOP,
                                                N_EPOCHS_STOP_FINAL, NUM_SAMPLES, dates, features,
                                                model_setup, test_data_sequence,
                                                train_data_sequence, val_data_sequence)
from src.tools import train_cgm

# Make sure correct model type
assert model_setup['type'] in ['gaussian', 'simple'], "Unknown model type!"

# Load score table if it already exists
try:
    scores = pd.read_csv(result_path / ('test_all_scores_' + model_setup['type'] + '.csv'))
except FileNotFoundError:
    scores = pd.DataFrame(columns=['model_id', 'RMSE', 'MARD', 'MAE',
                                   'A', 'B', 'C', 'D', 'E', 'precision', 'recall', 'F1', 'lag'])
    scores.index.name = '[training], val, test'

# Run through test scenarios
for i, (train_data, val_data, test_data) in enumerate(zip(train_data_sequence,
                                                          val_data_sequence,
                                                          test_data_sequence)):

    start_date_train = list(dates['start_date_train'][train_data])
    end_date_train = list(dates['end_date_train'][train_data])
    start_date_val = dates['start_date_val'][val_data]
    end_date_val = dates['end_date_val'][val_data]
    start_date_test = dates['start_date_test'][test_data]
    end_date_test = dates['end_date_test'][test_data]

    print("\n")
    print("--------------------------------------------")
    print('Case #{:d}'.format(i))
    print("TRAIN DATA:", train_data)
    print("VALIDATION DATA:", val_data)
    print("TEST DATA:", test_data)
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

    experiment_id = search_best_hypers(model_setup,
                                       num_samples=NUM_SAMPLES,
                                       n_epochs_stop=N_EPOCHS_STOP,
                                       max_num_epochs=MAX_NUM_EPOCHS,
                                       grace_period=GRACE_PERIOD,
                                       gpus_per_trial=0,
                                       data_obj=data_obj_hyperOpt)
    # experiment_id = main(num_samples=2, n_epochs_stop=3, max_num_epochs=2, gpus_per_trial=0, grace_period=1, data_obj=data_obj_hyperOpt)

    # %%
    print("\n")
    print("--------------------------------------------")
    print("Now retrain model with optimal parameters")
    exeriment_path = code_path / \
        'hyper_experiments' / (experiment_id + '.json')

    with open(exeriment_path) as json_file:
        experiment = json.load(json_file)

    best_model_dir = experiment['best_trial_dir']
    par_file = Path(best_model_dir) / '..' / 'params.json'

    # Build network and load weights
    with open(par_file) as json_file:
        optHyps = json.load(open(par_file))

    # Derive model type and define
    if model_setup['type'] == 'simple':
        model = DilatedNet(h1=optHyps["h1"], h2=optHyps["h2"], final_1x1_1=optHyps["final_1x1_1"], final_1x1_2=optHyps["final_1x1_2"])
    elif model_setup['type'] == 'gaussian':
        model = DilatedNetGaussian(h1=optHyps["h1"], h2=optHyps["h2"], final_1x1_1=optHyps["final_1x1_1"], final_1x1_2=optHyps["final_1x1_2"])

    data_obj = dataLoader(data_pars, features, n_steps_past=16,
                          n_steps_future=6,
                          allowed_gap=10,
                          scaler=StandardScaler())

    best_epoch_checkpoint_file_name = train_cgm(optHyps, model_setup, max_epochs=MAX_NUM_EPOCHS_FINAL,
                                                grace_period=GRACE_PERIOD_FINAL,
                                                n_epochs_stop=N_EPOCHS_STOP_FINAL,
                                                data_obj=data_obj_hyperOpt,
                                                use_ray_tune=False)
    # train_cgm(optHyps, max_epochs= 3, grace_period=1, n_epochs_stop=2, data_obj=data_obj, useRayTune=False)

    # Load best model state
    model_state, optimizer_state = torch.load(best_epoch_checkpoint_file_name)
    model.load_state_dict(model_state)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    user = getpass.getuser()
    model_id = f'id_{current_time}_{user}'

    model_figure_path = figure_path / model_id
    model_figure_path.mkdir(exist_ok=True, parents=True)

    best_model_path = model_path / model_id
    best_model_path.mkdir(exist_ok=True, parents=True)

    # Save best model
    copyfile(best_epoch_checkpoint_file_name,
             best_model_path / "final_model_checkpoint")

    # ---------------------------------------------------------------------
    # EVALUATE THE MODEL
    # ---------------------------------------------------------------------
    evaluationConfiguration = {
        'distance': 1,
        'hypo': 1,
        'clarke': 1,
        'parke': 1,
        'lag': 1,
        'plotLag': 1,
        'plotTimeseries': 1
    }
    # ---------------------------------------------------------------------

    # Define evaluation class
    evalObject = evaluateModel(data_obj_hyperOpt, model)

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

    scores.loc[str([train_data, val_data, test_data])] = [model_id,
                                                          distance['rmse'], distance['mard'], distance['mae'],
                                                          parkes_prob['A'], parkes_prob['B'], parkes_prob['C'], parkes_prob['D'], parkes_prob['E'],
                                                          hypo['precision'], hypo['recall'], hypo['F1'], lag
                                                          ]

    # Save results
    result_path.mkdir(exist_ok=True, parents=True)
    scores.to_csv(result_path / ('all_scores_' + model_setup['type'] + '.csv'))
    copyfile(par_file, model_figure_path / "optPars.json")
    copyfile(code_path / 'hyper_experiments' / (experiment_id +
                                                '.json'), model_figure_path / "data_properties.json")
