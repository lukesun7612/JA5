#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA5
@file: Feature selection.py
@time: 2021/5/25 19:51
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.model_selection import KFold
import statsmodels.api as sm
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance

def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    input1 = "./summary_Spain.xlsx"
    input2 = "./summary_China.csv"
    output1 = "./Table/Feature_selection1.xlsx"
    output2 = "./Table/Feature_selection2.xlsx"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID', engine='openpyxl')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    # df1 = df1.loc[df1['nearmiss_accel'].apply(lambda x: x > 0)]
    # df2 = df2.loc[df2['Harshacceleration'].apply(lambda x: x > 0)]
    df2['Fuel'] = df2['Fuel'].fillna(value=df2['Fuel'].mean())
    scaler = StandardScaler()
    X1 = df1.drop(columns=['speed_max', 'ndays', 'nearmiss_accel', 'nearmiss_brake', 'nearmiss_accel_under_50kmh',
                           'nearmiss_brake_above_120kmh'])
    X2 = df2[['Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
              'Trips', 'TripperDay', 'TripsinDay', 'DistanceperTrip', 'DistanceinDay', 'TripsinNight', 'DistanceinNight',
              'TripsinWeekdays', 'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
              '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']]
    name1, name2 = X1.columns.tolist(), X2.columns.tolist()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.fit_transform(X2)
    result = pd.DataFrame()
    fig1, ax = plt.subplots(2, 2)
    for m, y1 in enumerate(['nearmiss_accel', 'nearmiss_brake']):
        Y1 = df1[y1]

        poisson1 = TweedieRegressor(power=1, alpha=1, max_iter=10000).fit(X1, Y1, sample_weight=df1["ndays"]/7)
        poisson_gbrt1 = HistGradientBoostingRegressor(loss='poisson').fit(X1, Y1, sample_weight=df1["ndays"] / 7)
        poisson3 = GradientBoostingRegressor(loss='huber', random_state=0, n_iter_no_change=5).fit(X1, Y1, sample_weight=df1["ndays"] / 7)

        rfe1 = RFECV(poisson1, min_features_to_select=1, cv=KFold(10)).fit(X1, Y1)

        sfm1 = SelectFromModel(poisson1).fit(X1, Y1, sample_weight=df1["ndays"]/7)

        pi1 = permutation_importance(poisson1, X1, Y1, n_repeats=10, random_state=0, n_jobs=1)
        pi_gbrt1 =permutation_importance(poisson_gbrt1, X1, Y1, n_repeats=10, random_state=0, n_jobs=1)
        sorted_idx1 = pi1.importances_mean.argsort()
        sorted_idx2 = pi_gbrt1.importances_mean.argsort()
        ax[0][m].boxplot(pi1.importances[sorted_idx1].T, vert=False,
                         labels=np.array(name1)[sorted_idx1],
                         medianprops=dict(linewidth=0.5),
                         boxprops=dict(linewidth=0.5),
                         capprops=dict(linewidth=0.5),
                         whiskerprops=dict(linewidth=0.5),
                         flierprops=dict(markersize=0.5)
                         )
        ax[0][m].tick_params(labelsize=4)
        ax[0][m].set_title('Poisson'+'\n'+y1, fontsize=6)
        ax[1][m].boxplot(pi_gbrt1.importances[sorted_idx2].T, vert=False,
                         labels=np.array(name1)[sorted_idx2],
                         medianprops=dict(linewidth=0.5),
                         boxprops=dict(linewidth=0.5),
                         capprops=dict(linewidth=0.5),
                         whiskerprops=dict(linewidth=0.5),
                         flierprops=dict(markersize=0.5)
                         )
        ax[1][m].tick_params(labelsize=4)
        ax[1][m].set_title('GBRT'+'\n'+y1, fontsize=6)
        methodname = [[y1, y1, y1], ['REFCV', 'SelectFromModel', 'PermutationImportance']]
        result1 = pd.DataFrame(zip(rfe1.ranking_, sfm1.get_support(), map(lambda x: round(x, 4), pi1.importances_mean)), index=name1, columns=methodname)
        result = pd.concat([result, result1], axis=1)
        reg = Pipeline([("Feature selection", sfm1), ("regressor", HistGradientBoostingRegressor(loss='poisson'))]).fit(X1, Y1, regressor__sample_weight=df1["ndays"] / 7)
        print(mean_poisson_deviance(Y1, reg.predict(X1), sample_weight=df1["ndays"] / 7))
    result_ = pd.DataFrame()
    fig2, ax = plt.subplots(2, 2)
    for n, y2 in enumerate(['Harshacceleration', 'Harshdeceleration']):
        Y2 = df2[y2]

        poisson2 = TweedieRegressor(power=1, alpha=1, max_iter=10000).fit(X2, Y2, sample_weight=df2["Days"]/7)
        poisson_gbrt2 = HistGradientBoostingRegressor(loss='poisson').fit(X2, Y2, sample_weight=df2["Days"]/7)
        poisson3 = GradientBoostingRegressor(loss='huber', random_state=0, n_iter_no_change=5).fit(X2, Y2, sample_weight=df2["Days"] / 7)
        rfe2 = RFECV(poisson2, min_features_to_select=1, cv=KFold(10)).fit(X2, Y2)

        sfm2 = SelectFromModel(poisson2).fit(X2, Y2, sample_weight=df2["Days"]/7)

        pi2 = permutation_importance(poisson2, X2, Y2, n_repeats=10, random_state=0, n_jobs=1)
        pi_gbrt2 = permutation_importance(poisson_gbrt2, X2, Y2, n_repeats=10, random_state=0, n_jobs=1)
        sorted_idx1 = pi2.importances_mean.argsort()
        sorted_idx2 = pi_gbrt2.importances_mean.argsort()
        ax[0][n].boxplot(pi2.importances[sorted_idx1].T, vert=False,
                         labels=np.array(name2)[sorted_idx1],
                         medianprops=dict(linewidth=0.5),
                         boxprops=dict(linewidth=0.5),
                         capprops=dict(linewidth=0.5),
                         whiskerprops=dict(linewidth=0.5),
                         flierprops=dict(markersize=0.5)
                         )
        ax[0][n].tick_params(labelsize=4)
        ax[0][n].set_title('Poisson'+'\n'+y2, fontsize=6)
        ax[1][n].boxplot(pi_gbrt2.importances[sorted_idx2].T, vert=False,
                         labels=np.array(name2)[sorted_idx2],
                         medianprops=dict(linewidth=0.5),
                         boxprops=dict(linewidth=0.5),
                         capprops=dict(linewidth=0.5),
                         whiskerprops=dict(linewidth=0.5),
                         flierprops=dict(markersize=0.5)
                         )
        ax[1][n].tick_params(labelsize=4)
        ax[1][n].set_title('GBRT'+'\n'+y2, fontsize=6)
        methodname = [[y2, y2, y2],['REFCV', 'SelectFromModel', 'PermutationImportance']]
        result2 = pd.DataFrame(zip(rfe2.ranking_, sfm2.get_support(), map(lambda x: round(x, 4), pi2.importances_mean)), index=name2, columns=methodname)
        result_ = pd.concat([result_, result2], axis=1)
        reg = Pipeline([("Feature selection", sfm2), ("regressor", HistGradientBoostingRegressor(loss='poisson'))]).fit(X2, Y2, regressor__sample_weight=df2["Days"]/7)
        print(mean_poisson_deviance(Y2, reg.predict(X2), sample_weight=df2["Days"] / 7))
    fig1.suptitle("Spain Data", fontsize=8)
    fig2.suptitle("China Data", fontsize=8)
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
    # result.to_excel(output1)
    # result_.to_excel(output2)