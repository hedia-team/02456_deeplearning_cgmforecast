# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:20:49 2020

@author: madsobdrup
"""
import os

import pandas as pd

from .. import offsetSamplers

features = ['CGM', 'CHO', 'insulin', 'CGM_delta']

which_timeseries_form = 'Both'  # Choose between 'Both', 'CGM' or 'Difference'

# Paths to relevant folders relative to Hedia folder
train_data = ['575-ws-training']
val_data = '575-ws-training'
test_data = '575-ws-training'

# Define dataframe with start and end for different datasets
dates = pd.DataFrame(columns=['start_date_train', 'end_date_train',
                              'start_date_val', 'end_date_val',
                              'start_date_test', 'end_date_test'])

dates.loc['559-ws-training'] = ['2021-12-07 01:17:00', '2022-01-02 01:20:00',
                                '2022-01-02 01:25:00',  '2022-01-11 11:45:00',
                                '2022-01-11 11:50:00', '2022-01-17 23:56:00']

dates.loc['563-ws-training'] = ['2021-09-13 12:33:00', '2021-10-20 04:39:00',
                                '2021-10-20 04:44:00', '2021-10-24 18:17:00',
                                '2021-10-24 18:22:00', '2021-10-28 23:56:00']

dates.loc['570-ws-training'] = ['2021-12-07 16:29:00', '2022-01-08 20:18:00',
                                '2022-01-08 20:23:00', '2022-01-13 07:53:00',
                                '2022-01-13 07:58:00', '2022-01-16 23:59:00'
                                ]
dates.loc['575-ws-training'] = ['2021-11-17 12:04:00', '2021-12-23 16:25:00',
                                '2021-12-23 16:30:00', '2021-12-28 10:12:00',
                                '2021-12-28 10:17:00', '2022-01-01 23:55:00']

dates.loc['588-ws-training'] = ['2021-08-30 11:53:00', '2021-10-05 21:26:00',
                                '2021-10-05 21:31:00', '2021-10-10 11:00:00',
                                '2021-10-10 11:05:00', '2021-10-14 23:55:00']

dates.loc['591-ws-training'] = ['2021-11-30 17:06:00', '2022-01-05 22:36:00',
                                '2022-01-05 22:41:00', '2022-01-10 00:08:00',
                                '2022-01-10 00:13:00', '2022-01-13 23:58:00']


# Set label noise configurations
# Define label noise routines - Should probably de moved to parameters, but I have not succeeded yet
samplerCHO_train = offsetSamplers.constantSampler(0)
samplerInsulin_train = offsetSamplers.constantSampler(0)
samplerCHO_test = offsetSamplers.constantSampler(0)
samplerInsulin_test = offsetSamplers.constantSampler(0)

# Extract specified dates
start_date_train = list(dates['start_date_train'][train_data])
end_date_train = list(dates['end_date_train'][train_data])
start_date_val = dates['start_date_val'][val_data]
end_date_val = dates['end_date_val'][val_data]
start_date_test = dates['start_date_test'][test_data]
end_date_test = dates['end_date_test'][test_data]

# Parameters
seed = 1234  # For weitgh initilization
learning_rate = 0.0005
weight_decay = 0.001

max_epochs = 500
n_steps_future = 6  # Number of steps to predict into future
n_steps_past = 16  # 16 # Number necessary time steps back in time
batch_size_train = 12
batch_size_test = 64
dilations = [1, 1, 2, 4, 8]

parameter_file = os.path.basename(__file__)
