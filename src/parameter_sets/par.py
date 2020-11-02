# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:20:49 2020

@author: madsobdrup
"""
#%%

import numpy as np
import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from offsetSamplers import halfNormalSampler, constantSampler

features = ['CGM', 'CHO', 'insulin', 'CGM_delta']


which_timeseries_form = 'Both' # Choose between 'Both', 'CGM' or 'Difference'

# Paths to relevant folders relative to Hedia folder
train_data = ['575-ws-training']
test_data = '575-ws-training'

# Define dataframe with start and end for different datasets
dates = pd.DataFrame(columns = ['start_date_train', 'end_date_train', 'start_date_test', 'end_date_test'])

dates.loc['559-ws-training'] = ['2021-12-07 01:17:00', '2022-01-05 23:56:00', '2022-01-06 00:01:00', '2022-01-17 23:56:00']
dates.loc['563-ws-training'] = ['2021-09-13 12:33:00', '2021-10-13 19:45:00', '2021-10-13 19:50:00', '2021-10-28 23:56:00']
dates.loc['575-ws-training'] = ['2021-11-17 12:04:00', '2021-12-17 21:57:00', '2021-12-17 22:02:00', '2022-01-01 23:55:00']
dates.loc['570-ws-training'] = ['2021-12-07 16:29:00', '2022-01-06 00:11:00', '2022-01-06 00:16:00', '2022-01-16 23:59:00']
dates.loc['588-ws-training'] = ['2021-08-30 11:53:00', '2021-10-03 02:43:00', '2021-10-03 02:48:00', '2021-10-14 23:55:00']
dates.loc['adult#001'] =       ['2019-09-16 00:00:00', '2020-01-17 06:55:00', '2020-01-17 07:00:00', '2020-03-14 00:00:00']
# Set label noise configurations

# Define label noise routines - Should probably de moved to parameters, but I have not succeeded yet
samplerCHO_train = constantSampler(0)
samplerInsulin_train = constantSampler(0)
samplerCHO_test = constantSampler(0)
samplerInsulin_test = constantSampler(0)

# Extract specified dates
start_date_train = list(dates['start_date_train'][train_data])
end_date_train = list(dates['end_date_train'][train_data])
start_date_test = dates['start_date_test'][test_data]
end_date_test = dates['end_date_test'][test_data]

# Parameters
seed = 1234 # For weitgh initilization
learning_rate = 0.0005
weight_decay = 0.001

max_epochs = 500
n_steps_future = 6 # Number of steps to predict into future
n_steps_past = 16 #16 # Number necessary time steps back in time
batch_size_train = 12
batch_size_test = 64
dilations = [1,1,2,4,8]


# Seed to achieve same resuls when running same model
seed = 1234


parameter_file = os.path.basename(__file__)

# Plot loss along the way
running_plot = False
