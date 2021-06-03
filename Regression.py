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
from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance

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
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(f"WARNING: Estimator yields invalid, non-positive predictions "
              f" for {n_masked} samples out of {n_samples}. These predictions "
              f"are ignored when computing the Poisson deviance.")

    print("mean Poisson deviance: %.3f" %
          mean_poisson_deviance(Y[mask],
                                y_pred[mask],
                                sample_weight=E[mask]))

if __name__ == '__main__':
    scaler = StandardScaler()
    input1 = "./summary_drivers.xlsx"
    input2 = "./summary0.csv"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID', engine='openpyxl')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    df1.rename(columns={'ndays': 'Days'}, inplace=True)
    # df1 = df1.loc[df1['nearmiss_accel'].apply(lambda x: x > 0)]
    # df2 = df2.loc[df2['Harshacceleration'].apply(lambda x: x > 0)]
    df2['Fuel'] = df2['Fuel'].fillna(value=df2['Fuel'].mean())
    X1 = df1[['ntrips_per_day_media', 'nweekdays', 'nweekenddays', 'ntrips_under_15min', 'distance_week_media', 'distance_per_trip_media', 'week', 'weekend', 'distance_under_50kmh']]
    X2 = df1[['headig_mean', 'ntrips_per_day_media', 'nweekdays', 'nweekenddays', 'ntrips_under_15min', 'ntrips_between_15min_30min', 'ntrips_between_30min_1h', 'distance_week_media', 'weekend', 'distance_during_night', 'distance_under_50kmh','distance_above_120kmh']]
    X3 = df2[['Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'TripperDays', 'DistanceinNight', 'TripsinNight', 'TripsinWeekends', 'Trips<15m', '30m<Trips<1h', '1h<Trips<2h']]
    X4 = df2[['Fuel', 'Brakes', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate', 'TripsinNight', '30m<Trips<1h']]
    Y1 = df1['nearmiss_accel']
    Y2 = df1['nearmiss_brake']
    Y3 = df2['Harshacceleration']
    Y4 = df2['Harshdeceleration']
    for n, (X, Y, E) in enumerate(zip([X1, X2, X3, X4], [Y1, Y2, Y3, Y4], [df1['Days'], df1['Days'], df2['Days'], df2['Days']])):
        E = E/7
        p = 1
        # if n < 2:
        #     p = 1.4
        # else:
        #     p = 1.5
        X = scaler.fit_transform(X)
        X = sm.add_constant(X)
        poisson1 = sm.GLM(Y.values.tolist(), X, family=sm.families.NegativeBinomial(sm.families.links.log()), exposure=E.values.tolist(), missing='drop').fit() #cov_type='HC0'
        poisson2 = PoissonRegressor(fit_intercept=False, max_iter=1000).fit(X, Y, sample_weight=E)
        poisson3 = TweedieRegressor(power=p, alpha=1, fit_intercept=False, max_iter=1000).fit(X, Y, sample_weight=E)
        poisson_gbrt = HistGradientBoostingRegressor(loss='poisson').fit(X, Y, sample_weight=E)
        n += 1
        # print(poisson1.summary())
        print("model1")
        # score_estimator(poisson1, X)
        print("model2")
        score_estimator(poisson2, X)
        print("model3")
        score_estimator(poisson3, X)
        print("model4")
        score_estimator(poisson_gbrt, X)
        print(poisson_gbrt.score(X, Y))