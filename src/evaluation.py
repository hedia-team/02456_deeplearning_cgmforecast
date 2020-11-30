# %%

from collections.abc import Iterable
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import cm
from plotly.offline import plot
from plotly.subplots import make_subplots
from torch import nn

import src.metrics as metrics
from src.ClarkeErrorGrid.ClarkeErrorGrid import clarke_error_grid
from src.data import DataframeDataLoader
from src.ParkesErrorGrid.ParkesErrorGrid import parke_error_grid
from src.tools import crosscorr, predict_cgm


class evaluateModel:
    def __init__(self, dataObject, model: nn.Module):

        # Extract test data from data obejct
        self.test_set = dataObject.load_test_data()

        self.n_elements = count_iterable(self.test_set)
        self.test_loader = DataframeDataLoader(
            self.test_set,
            batch_size=self.n_elements,
            shuffle=False,
        )
        self.test_df = self.test_loader.sample_dataframe
        self.test_predictions = predict_cgm(dataObject, model)

        self.test_predictions_delta = self.test_predictions[:, 0]

        # Save standard deviation predictions if possible
        try:
            self.test_predictions_std = self.test_predictions[:, 1]
        except:
            self.test_predictions_std = None

        self.test_predictions_absolute = self.test_predictions_delta + \
            self.test_df['CGM']

    def get_distanceAnalysis(self, dataType='test'):

        # Determine which part of the data object that should be used
        if dataType == 'test':
            _pred = self.test_predictions_absolute
            _target = self.test_df['target']

        elif dataType == 'train':
            pass
        elif dataType == 'validation':
            pass

        # Compute and save distance measures
        _d = {}
        _d['rmse'], _d['mard'], _d['mae'], _d['mape'] = metrics.report(
            _pred, _target)
        return _d

    def get_hypoAnalysis(self, dataType='test'):

        # Determine which part of the data object that should be used
        if dataType == 'test':
            _pred = self.test_predictions_absolute
            _target = self.test_df['target']

        elif dataType == 'train':
            pass
        elif dataType == 'validation':
            pass

        # Compute confusion values
        _d = {}
        _d['recall'], _d['precision'], _d['F1'], _d['tn'], _d['fp'], _d['fn'], _d['tp'] = metrics.confusion_hypo(
            _pred, _target)
        return _d

    def get_lagAnalysis(self, dataType='test', figure_path=None):

        # Determine which part of the data object that should be used
        if dataType == 'test':
            _pred = self.test_predictions_absolute
            _target = self.test_df['target']

        elif dataType == 'train':
            pass
        elif dataType == 'validation':
            pass

        _d = {}
        _d['max_lag_time'], _d['corr'] = identifyLag(
            _pred, _target, figure_path)
        return _d

    def get_timeSeriesPlot(self, figure_path=None, dataType='test'):

        # Determine which part of the data object that should be used
        if dataType == 'test':
            _pred = self.test_predictions_absolute
            _std = self.test_predictions_std
            _target = self.test_df

        elif dataType == 'train':
            pass
        elif dataType == 'validation':
            pass

        # Run plotting function
        plotPredictionTimeseries(_pred, _std, _target, figure_path)

    def apply_clarkes_error_grid(self, unit, figure_path=None, dataType='test') -> Tuple[dict, dict]:

        # Determine which part of the data object that should be used
        if dataType == 'test':
            _pred = self.test_predictions_absolute
            _target = self.test_df['target']

        elif dataType == 'train':
            pass
        elif dataType == 'validation':
            pass

        figure, zones, zones_prob = clarke_error_grid(_target, _pred, '', unit)

        if figure_path is not None:
            figure.savefig(figure_path / 'clarke.png')
            figure.savefig(figure_path / 'clarke.pdf', format='pdf')

        return zones, zones_prob

    def apply_parkes_error_grid(self, unit, figure_path=None, dataType='test') -> Tuple[dict, dict]:

        # Determine which part of the data object that should be used
        if dataType == 'test':
            _pred = self.test_predictions_absolute
            _target = self.test_df['target']

        elif dataType == 'train':
            pass
        elif dataType == 'validation':
            pass

        figure, zones, zones_prob = parke_error_grid(np.array(_target), np.array(_pred), '', unit)

        if figure_path is not None:
            figure.savefig(figure_path / 'parke.png')
            figure.savefig(figure_path / 'parke.pdf', format='pdf')

        return zones, zones_prob


def count_iterable(i: Iterable):
    return sum(1 for e in i)


def identifyLag(predictions: np.array, targets: np.array, figure_path=None) -> Tuple[float, float]:
    d1 = targets
    d2 = pd.Series(predictions)
    crosscorr(d1, d2, lag=-6, wrap=False)
    lags = range(-int(10), 1)
    rs = [crosscorr(d1, d2, lag) for lag in lags]
    max_corr = np.argmax(rs)

    if figure_path is not None:
        f, ax = plt.subplots(figsize=(14, 3))
        ax.plot(lags, rs)
        ax.axvline(lags[max_corr], color='r',
                   linestyle='--', label='Peak synchrony')
        ax.set_xticks(list(lags))
        ax.set_xticklabels(list([lags[i] * 5 for i in range(len(lags))]))
        ax.legend()
        ax.grid()

        #f.savefig(model_figure_path / 'autocorr.png')
        f.savefig(figure_path / 'autocorr.pdf', format='pdf')

    return str(-1*lags[max_corr] * 5), rs[max_corr]


def plotPredictionTimeseries(predictions: np.array, standard_dev: np.array,  cgm_df: pd.DataFrame, figure_path: str):

    predictions_delta = predictions - cgm_df['CGM']

    sectionIdx = np.append(np.insert(
        1+np.where((cgm_df.index[1:] - cgm_df.index[:-1]) > pd.Timedelta('6 minutes'))[0], 0, 0), len(cgm_df))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for section in range(len(sectionIdx)-1):
        sectionRange = range(sectionIdx[section], sectionIdx[section+1])
        sectionDates = cgm_df.index[(
            sectionIdx[section]):sectionIdx[section+1]]

        if standard_dev is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(sectionDates),
                    y=list(predictions[sectionDates] / 18 + standard_dev[sectionRange] / 18),
                    #ymax=list(predictions[sectionDates] / 18 + standard_dev[sectionRange] / 18),
                    mode='lines',
                    name='Standard Deviation',
                    line_color='rgba' + str(cm.Blues(80)),
                    legendgroup='std',
                    line_dash='dash',
                    showlegend=(section == 0),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(sectionDates),
                    y=list(predictions[sectionDates] / 18 - standard_dev[sectionRange] / 18),
                    #ymax=list(predictions[sectionDates] / 18 + standard_dev[sectionRange] / 18),
                    mode='lines',
                    fill='tonexty',
                    name='Standard Deviation',
                    line_color='rgba' + str(cm.Blues(80)),
                    line_dash='dash',
                    legendgroup='std',
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=list(sectionDates),
                y=list(predictions[sectionDates] / 18),
                name='Prediction',
                marker_color='rgba' + str(cm.Blues(300)),
                legendgroup='prediction',
                showlegend=(section == 0)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(sectionDates),
                y=list(predictions_delta[sectionDates] / 18),
                name='Prediction delta',
                marker_color='rgba' + str(cm.Blues(300)),
                legendgroup='prediction_delta',
                showlegend=(section == 0)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(sectionDates),
                y=list(cgm_df['target'][sectionDates] / 18),
                name="True",
                marker_color='rgba' + str(cm.Blues(100)),
                legendgroup='target',
                showlegend=(section == 0)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(sectionDates),
                y=list(cgm_df['target_delta'][sectionDates] / 18),
                name="True delta",
                marker_color='rgba' + str(cm.Blues(100)),
                legendgroup='target_delta',
                showlegend=(section == 0)
            )
        )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1d",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1wk",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1mth",
                         step="month",
                         stepmode="backward")
                ]),
                yanchor="top",
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        #title="Model: " + str(model_id) + ", Train: " + str(train_data) + ",<br> Test: " + test_data,
        xaxis_title="Time",
        yaxis_title="Blood Glucose Measurement [Mmol/L]",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        yaxis=dict(range=[-5, 30]),
    )
    fig.add_trace(
        go.Scatter(
            x=[cgm_df.index.min(), cgm_df.index.max()],
            y=[12, 12],
            name="Upper range",
            marker_color='black',
            mode='lines',
            line=dict(dash='dash', width=1)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[cgm_df.index.min(), cgm_df.index.max()],
            y=[4, 4],
            name="Lower range",
            marker_color='black',
            mode='lines',
            line=dict(dash='dash', width=1)
        )
    )

    for i, feature in enumerate(['CHO', 'insulin']):
        df = cgm_df.loc[cgm_df[feature] != 0]
        fig.add_trace(
            go.Scatter(
                x=list(df.index),
                y=list(df[feature]),
                name=str(feature),
                marker_color='rgba' + str(cm.Reds(100 + 200 * (i - 1))),
                mode='markers'),
            secondary_y=True
        )

    plot(fig, filename=str(figure_path / 'predictionPlot.html'))
