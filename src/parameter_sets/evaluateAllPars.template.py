import pandas as pd

features = ['CGM', 'CHO', 'insulin', 'CGM_delta']

# Define search parameters
# ----------------------------------------
NUM_SAMPLES = 40  # Number of different hyper parameter setting
MAX_NUM_EPOCHS = 50  # Maximum number of epochs in training
N_EPOCHS_STOP = 15  # Number of consecuetive epochs with no improvement in validation data before terminating training
GRACE_PERIOD = 5  # Minimum number of epochs before termination is allowed
# ----------------------------------------

# Define final train parameters
# ----------------------------------------
MAX_NUM_EPOCHS_FINAL = 50
N_EPOCHS_STOP_FINAL = 15
GRACE_PERIOD_FINAL = 5
# ----------------------------------------


# Define dates
dates = pd.DataFrame(
    columns=['start_date_train', 'end_date_train',
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

# Define data set
train_data_sequence = [['575-ws-training'],
                       ['570-ws-training'],
                       ['563-ws-training'],
                       ['559-ws-training'],
                       ['591-ws-training'],
                       ['588-ws-training'],
                       ['575-ws-training', '559-ws-training', '570-ws-training', '559-ws-training', '591-ws-training', '588-ws-training'],
                       ['575-ws-training', '559-ws-training', '570-ws-training', '559-ws-training', '591-ws-training', '588-ws-training'],
                       ['575-ws-training', '559-ws-training', '570-ws-training', '559-ws-training', '591-ws-training', '588-ws-training'],
                       ['575-ws-training', '559-ws-training', '570-ws-training', '559-ws-training', '591-ws-training', '588-ws-training'],
                       ['575-ws-training', '559-ws-training', '570-ws-training', '559-ws-training', '591-ws-training', '588-ws-training'],
                       ['575-ws-training', '559-ws-training', '570-ws-training', '559-ws-training', '591-ws-training', '588-ws-training']
                       ]

val_data_sequence = ['575-ws-training',
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
