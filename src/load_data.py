import pandas as pd

from src.data import DataframeDataset, DataframeDataLoader, ConcatDataframeDataset, CreateDatasetWrapper


class dataLoader:
    def __init__(self, data_pars, features, n_steps_past, n_steps_future, allowed_gap, scaler):

        
        self.features = features
        self.n_steps_past = n_steps_past
        self.n_steps_future = n_steps_future
        self.allowed_gap = allowed_gap
        self.scaler = scaler

        self.data_path = data_pars['path']
        self.train_data = data_pars['train_data']
        self.test_data = data_pars['test_data']
        self.validation_data = data_pars['validation_data']

        self.start_date_train = data_pars['start_date_train']
        self.start_date_test = data_pars['start_date_test']
        self.start_date_validation = data_pars['start_date_validation']

        self.end_date_train = data_pars['end_date_train']
        self.end_date_test = data_pars['end_date_test']
        self.end_date_validation = data_pars['end_date_validation']
                 
    def load_test_data(self, test_data=None, start_date_test=None, end_date_test=None):
        if test_data is None:
            test_data = self.test_data
            start_date_test = self.start_date_test
            end_date_test = self.end_date_test

        # Prepare test se
        test_file = self.data_path / f'{test_data}.csv'
        df = pd.read_csv(
            test_file,
            parse_dates=['Time'],
            infer_datetime_format=True,
            index_col='Time',
        )

        df = df.loc[start_date_test:end_date_test].copy()

        dset_test = CreateDatasetWrapper(
            df=df,
            features=self.features,
            n_steps_past=self.n_steps_past,
            n_steps_future=self.n_steps_future,
            allowed_gap=self.allowed_gap,
            scaler=self.scaler,#scaler.fit_transform if i == 0 else scaler.transform
            fit=True,
            skip_missing_data=True
            )

        return dset_test


    def load_train_and_val(self):
        # Prepare training set
        #scaler = MinMaxScaler()
        train_datasets = []
        for i in range(len(self.train_data)):
            df = pd.read_csv(
                self.data_path / f'{self.train_data[i]}.csv',
                parse_dates=['Time'],
                infer_datetime_format=True,
                index_col='Time',
            )
            df = df.loc[self.start_date_train[i]:self.end_date_train[i]].copy()

            train_datasets.append(
                CreateDatasetWrapper(
                    df=df,
                    features=self.features,
                    n_steps_past=self.n_steps_past,
                    n_steps_future=self.n_steps_future,
                    allowed_gap=self.allowed_gap,
                    scaler=self.scaler,#scaler.fit_transform if i == 0 else scaler.transform
                    fit=True if i == 0 else False,
                    skip_missing_data=True
                )
            )
        train_datasets_concat = ConcatDataframeDataset(train_datasets)
            

        dset_val = self.load_test_data(self.validation_data, self.start_date_validation, self.end_date_validation)

        return train_datasets_concat, dset_val