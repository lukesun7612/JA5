#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA5
@file: Regression.py
@time: 2021/5/27 1:36
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance, explained_variance_score, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
def score_estimator(estimator, X):
    """Score an estimator on the test set."""
    y_pred = estimator.predict(X)

    print("MSE: %.3f" %
          mean_squared_error(Y, y_pred,
                             sample_weight=E))
    print("MAE: %.3f" %
          mean_absolute_error(Y, y_pred,
                              sample_weight=E))

    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance.
    mask = y_pred >= 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(f"WARNING: Estimator yields invalid, non-positive predictions "
              f" for {n_masked} samples out of {n_samples}. These predictions "
              f"are ignored when computing the Poisson deviance.")

    print("mean Poisson deviance: %.3f" %
          mean_poisson_deviance(Y,
                                y_pred,
                                sample_weight=E))
    print("explained variance score: %.3f" %explained_variance_score(Y, y_pred, sample_weight=E))
    print("r2 score: %.3f" %r2_score(Y, y_pred, sample_weight=E))

if __name__ == '__main__':
    scaler = StandardScaler()
    input1 = "./summary_Spain.xlsx"
    input2 = "./summary_China.csv"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID', engine='openpyxl')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    df1.rename(columns={'ndays': 'Days'}, inplace=True)
    # df1 = df1.loc[df1['nearmiss_accel'].apply(lambda x: x > 0)]
    # df2 = df2.loc[df2['Harshacceleration'].apply(lambda x: x > 0)]
    df2['Fuel'] = df2['Fuel'].fillna(value=df2['Fuel'].mean())
    X1 = df1.drop(columns=['speed_max', 'Days', 'nearmiss_accel', 'nearmiss_brake', 'nearmiss_accel_under_50kmh',
                           'nearmiss_brake_above_120kmh'])
    X2 = df2[['Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
              'Trips', 'TripperDay', 'TripsinDay', 'DistanceperTrip', 'DistanceinDay', 'TripsinNight',
              'DistanceinNight',
              'TripsinWeekdays', 'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m',
              '15m<Trips<30m',
              '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']]
    name1, name2 = X1.columns.tolist(), X2.columns.tolist()
    Y1 = df1['nearmiss_accel']
    Y2 = df1['nearmiss_brake']
    Y3 = df2['Harshacceleration']
    Y4 = df2['Harshdeceleration']
    n_bins = 100
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(16, 4))
    # fig.subplots_adjust(bottom=0.2)
    ax0.set_title("China Data")
    _ = df2["Harshacceleration"].hist(bins=n_bins, log=True, ax=ax0)
    ax0.set_xlabel("Harshacceleration")
    ax1.set_title("China Data")
    _ = df2["Harshdeceleration"].hist(bins=n_bins, log=True, ax=ax1)
    ax1.set_xlabel("Harshdeceleration")
    ax2.set_title("Spain Data")
    _ = df1["nearmiss_accel"].hist(bins=n_bins, log=True, ax=ax2)
    ax2.set_xlabel("nearmiss_accel")
    ax3.set_title("Spain Data")
    _ = df1["nearmiss_brake"].hist(bins=n_bins, log=True, ax=ax3)
    ax3.set_xlabel("nearmiss_brake")
    ax0.set_ylabel("Frequency")
    # plt.show()

    for n, (X, Y, E, label) in enumerate(zip([X1, X1, X2, X2], [Y1, Y2, Y3, Y4], [df1['Days'], df1['Days'], df2['Days'], df2['Days']], ['Spain Data', 'Spain Data', 'China Data', 'China Data'])):
        E = E/7
        X = scaler.fit_transform(X)
        # X = sm.add_constant(X)
        # poisson1 = sm.GLM(Y.values.tolist(), X, family=sm.families.Poisson(sm.families.links.log()), exposure=E.values.tolist(), missing='drop').fit() #cov_type='HC0'
        poisson2 = PoissonRegressor(fit_intercept=True, max_iter=1000).fit(X, Y, sample_weight=E)
        poisson3 = GradientBoostingRegressor(loss='huber', random_state=0, n_iter_no_change=5).fit(X, Y, sample_weight=E)
        poisson_gbrt = HistGradientBoostingRegressor(loss='poisson').fit(X, Y, sample_weight=E)
        # rfe1 = RFECV(poisson2, min_features_to_select=1, cv=10).fit(X1, Y1)
        sfm1 = SelectFromModel(poisson2).fit(X, Y, sample_weight=E)
        sfm2 = SelectFromModel(poisson3).fit(X, Y, sample_weight=E)
        reg1 = Pipeline([("Feature selection", sfm1), ("regressor", PoissonRegressor(fit_intercept=False, max_iter=1000))]).fit(
            X, Y, regressor__sample_weight=E)
        reg2 = Pipeline([("Feature selection", sfm2), ("regressor", GradientBoostingRegressor(loss='huber', random_state=0, n_iter_no_change=5))]).fit(
            X, Y, regressor__sample_weight=E)
        reg3 = Pipeline([("Feature selection", sfm2), ("regressor", HistGradientBoostingRegressor(loss='poisson'))]).fit(
            X, Y, regressor__sample_weight=E)
        # print(poisson1.summary())
        # print("model1")
        # score_estimator(poisson1, X)

        print("Poisson")
        score_estimator(poisson2, X)
        print(poisson2.score(X, Y))

        print("Poisson_selected")
        score_estimator(reg1, X)
        print(reg1.score(X, Y))

        print("GBRT")
        score_estimator(poisson3, X)
        print(poisson3.score(X, Y))

        print("GBRT_SELECTED")
        score_estimator(reg2, X)
        print(reg2.score(X, Y))

        print("HGBRT")
        score_estimator(poisson_gbrt, X)
        print(poisson_gbrt.score(X, Y))

        print("HGBRT_SELECTED")
        score_estimator(reg3, X)
        print(reg3.score(X, Y))



        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), sharey=True)
        # fig.subplots_adjust(bottom=0.2)
        Y.hist(bins=n_bins, ax=axes[0])
        axes[0].set_title(label)
        axes[0].set_yscale('log')
        axes[0].set_xlabel(Y.name)
        # axes[row_idx, 0].set_ylim([1e1, 5e5])
        axes[0].set_ylabel("Frequency")

        for idx, model in enumerate([reg1, reg2, poisson_gbrt]):
            y_pred = model.predict(X)

            pd.Series(y_pred).hist(bins=n_bins,
                                       ax=axes[idx + 1])
            axes[idx + 1].set(
                    # title=model[-1].__class__.__name__,
                    yscale='log',
                    xlabel=Y.name + "(predicted)"
                )
    plt.tight_layout()
    plt.show()