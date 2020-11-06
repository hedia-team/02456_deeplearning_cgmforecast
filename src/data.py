"""
Project:    gresearch
File:       data.py
Created by: mads
On:         03/02/20
At:         4:56 PM
"""
from typing import List, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset


class DataframeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, features: List[str], n_steps_past: int, n_steps_future: int,
                 transform=None):
        """

        :param dataframe:
        :param features: The column labels to include as input for the
            model.
        :param n_steps_past: the number of previous CGM readings to
            include in a sample including the current newest reading.
        :param n_steps_future: How many steps into the future should the
            target be?
        """
        assert transform is not None

        self.n_steps_past = n_steps_past
        self.n_steps_future = n_steps_future

        dataframe = fill_missing_values(dataframe, dataframe.columns)

        # Add dynamically computed features
        for feature in features:
            if feature not in dataframe.columns:
                if feature[-6:] == '_delta':
                    base_feature = feature[:-6]
                    dataframe[feature] = dataframe[base_feature].diff(1)
                else:
                    raise ValueError(f'The feature {feature} does not exist in the dataframe and it doesnt match a'
                                     f'pattern for being generated automatically.')

        # Prepare targets
        dataframe['target'] = dataframe['CGM'].shift(-n_steps_future)  # The absolute BGL value we try to predict
        dataframe['target_delta'] = dataframe['CGM'].diff(n_steps_future).shift(-n_steps_future)  # delta now to then
        self.dataframe = dataframe  # Store the result (including NaNs)

        # Remove NaNs from diffing and shifting
        self.feed_df = dataframe.dropna(subset=['CGM_delta', 'target', 'target_delta'])
        self.feed_df_transformed = pd.DataFrame(transform(self.feed_df[features].values),columns = features, index = self.feed_df.index)

        # Prepare Torch tensors
        self.input_feature_tensor = torch.tensor(self.feed_df_transformed.values, dtype=torch.float32)
        self.target_tensor = torch.tensor(self.feed_df['target_delta'].values, dtype=torch.float32)

    def __getitem__(self, index):
        input_features = self.input_feature_tensor[index:index + self.n_steps_past, :]
        target = self.target_tensor[index + self.n_steps_past - 1]

        return input_features, target

    def __len__(self):

        return len(self.feed_df) - (self.n_steps_past - 1)

    @property
    def sample_dataframe(self) -> pd.DataFrame:
        """Get the a dataframe where each row corresponds to a sample in the dataset without past timeseries data."""
        return self.feed_df.iloc[self.n_steps_past-1:]

    @property
    def sample_dataframe_transformed(self):
        """Get the a dataframe where each row corresponds to a sample in the dataset without past timeseries data."""
        return self.feed_df_transformed.iloc[self.n_steps_past-1:]

    def sample_dataframe_batch_sized(self, batch_size: int):
        return self.sample_dataframe[:-(len(self.sample_dataframe) % batch_size)]


class ConcatDataframeDataset(ConcatDataset):
    datasets: List[DataframeDataset]

    @property
    def sample_dataframe(self):
        """Get the a dataframe where each row corresponds to a sample in the dataset without past timeseries data."""
        return pd.concat([item.sample_dataframe for item in self.datasets])

    @property
    def sample_dataframe_transformed(self):
        """Get the a transformed dataframe where each row corresponds to a sample in the dataset without past timeseries data."""
        return pd.concat([item.sample_dataframe_transformed for item in self.datasets])


class DataframeDataLoader(DataLoader):
    dataset: Union[DataframeDataset, ConcatDataframeDataset]

    @property
    def sample_dataframe(self):
        """Get the a dataframe where each row corresponds to a sample in the dataset without past timeseries data.
        The dataframe will be sliced to match with the data that is retrieved when iterating over this dataloader.
        """
        if self.drop_last:
            end_index = len(self.dataset.sample_dataframe) // self.batch_size * self.batch_size
            return self.dataset.sample_dataframe.iloc[:end_index]
        else:
            return self.dataset.sample_dataframe


class SimulatedData(Dataset):
    def __init__(self, filename, n_steps_past=16, inputs=['CGM'], start_date='2019-09-16',
                 end_date='2020-03-14', n_steps_future=6, which_timeseries_form='Both', transform=None):
        """

        :param folder_dataset: str
        :param T: int
        :param symbols: list of str
        :param use_columns: bool
        :param start_date: str, date format YYY-MM-DD
        :param end_date: str, date format YYY-MM-DD
        """
        self.inputs = inputs
        if len(inputs) == 0:
            print("No inputs was specified")
            return
        self.start_date = start_date
        if len(start_date) == 0:
            print("No start date was specified")
            return
        self.end_date = end_date
        if len(end_date) == 0:
            print("No end date was specified")
            return
        self.n_steps_past = n_steps_past
        self.n_steps_future = n_steps_future

        self.use_columns = ['Time', 'CGM']
        self.use_columns.extend(x for x in inputs if x not in self.use_columns)

        self.which_timeseries_form = which_timeseries_form

        df = pd.read_csv(
            filename,
            usecols=self.use_columns,
            parse_dates=['Time'],
            infer_datetime_format=True,
            index_col='Time',
            na_values='nan'
        )

        df = df.loc[start_date:end_date].copy()

        # Handle missing values
        self.df_data = fill_missing_values(df, self.inputs)

        # Add one step differences
        self.df_data['one_step_diff'] = self.df_data['CGM'].diff(1)
        self.df_data = self.df_data[1:]

        self.numpy_data = self.df_data.values
        if transform is None:
            print("WARNING: no transform set!")
            self.train_data = self.numpy_data
        else:
            self.train_data = transform(self.numpy_data)

        self.chunks = torch.FloatTensor(self.train_data).unfold(0, self.n_steps_past + self.n_steps_future, 1).permute(
            0, 2, 1)

    def __getitem__(self, index):
        if self.which_timeseries_form == 'Both':
            input = self.chunks[index, :self.n_steps_past, :]
        elif self.which_timeseries_form == 'CGM':
            input = self.chunks[index, :self.n_steps_past, :-1]
        elif self.which_timeseries_form == 'Difference':
            input = self.chunks[index, :self.n_steps_past, 1:]

        last_observation = self.chunks[index, self.n_steps_past-1, :]
        target = self.chunks[index, -1, :] - last_observation
        return input, target, last_observation

    def __len__(self):
        return self.chunks.size(0)


def fill_missing_values(df, columns, allowed_gap=None, fill_cgm_na=-9999):
    for column in columns:
        if column == 'CGM':  # Interpolate nan from rest of data
            if allowed_gap == None:
                df['CGM'] = df['CGM'].interpolate(method='polynomial', order=1)
            else:
                # Interpolate nans away when nan-periods are shorter than allowed_gap
                mask = df.copy()
                grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
                grp['ones'] = 1
                mask = (grp.groupby('CGM')['ones'].transform('count') < allowed_gap) | df['CGM'].notnull()

                cgm_inter= df['CGM'].interpolate(method='polynomial', order=1).copy()
                df.loc[mask, 'CGM'] = cgm_inter[mask]

                # Find all block of valid data (assuming that data does not begin or end)
                df['CGM'] = df['CGM'].fillna(fill_cgm_na)

        else:  # Set all nan to zero
            df[column] = df[column].fillna(0)

    return df


def CreateDatasetWrapper(df: pd.DataFrame, features: List[str], n_steps_past: int, n_steps_future: int,
        allowed_gap: int, scaler=None, fit=True, skip_missing_data=True):

    assert scaler is not None

    # check that data does not begin or end with nan
    assert np.invert(np.isnan(df['CGM'].iloc[0]))
    assert np.invert(np.isnan(df['CGM'].iloc[-1]))

    if skip_missing_data:
        
        df = fill_missing_values(df, df.columns, allowed_gap=allowed_gap, fill_cgm_na=-9999)
        starts_nan = np.where(df['CGM'].diff(1) < -1000)[0]
        ends_nan = np.where(df['CGM'].diff(1) > 1000)[0]

        ends_valid = np.append(starts_nan,len(df['CGM'])-1) 
        starts_valid = np.insert(ends_nan,0,0)


        # Run through all blocks and create full dataset to fit transformation on
        if fit:
            all_dsets = []
            for i in range(len(starts_valid)):
                subdf =df.iloc[starts_valid[i]:ends_valid[i]].copy()
                if subdf.shape[0] > (n_steps_past  + n_steps_future):
                    dset = DataframeDataset(
                                dataframe=subdf,
                                features=features,
                                n_steps_past=n_steps_past,
                                n_steps_future=n_steps_future,
                                transform=scaler.fit_transform  if fit else scaler.transform# not hasattr(scaler,'n_samples_seen_') else scaler.transform 
                        )

                    all_dsets.append(dset.sample_dataframe)

            # Fit the transformer
            scaler.fit_transform(pd.concat(all_dsets)[features])

        # Create dataset using the fitted transformation
        train_datasets = []
        for i in range(len(starts_valid)):
            subdf=df.iloc[starts_valid[i]:ends_valid[i]].copy()
            if subdf.shape[0] > (n_steps_past  + n_steps_future):
                train_datasets.append(
                        DataframeDataset(
                            dataframe=subdf,
                            features=features,
                            n_steps_past=n_steps_past,
                            n_steps_future=n_steps_future,
                            transform=scaler.transform
                        )
                    )
            train_datasets_concat_singlecsv = ConcatDataframeDataset(train_datasets)

        return train_datasets_concat_singlecsv

    else:
        dset = DataframeDataset(
            dataframe=df,
            features=features,
            n_steps_past=n_steps_past,
            n_steps_future=n_steps_future,
            transform=scaler.fit_transform if fit else scaler.transform
        )

        return dset