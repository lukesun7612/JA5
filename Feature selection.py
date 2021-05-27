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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import TweedieRegressor, LassoCV, PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance

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
    input1 = "./summary_drivers.xlsx"
    input2 = "./summary0.csv"
    output1 = "./Table/Feature_selection1.xlsx"
    output2 = "./Table/Feature_selection2.xlsx"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    df1 = df1.loc[df1['nearmiss_accel'].apply(lambda x: x > 0)]
    # df2 = df2.loc[df2['Harshacceleration'].apply(lambda x: x > 0)]
    df2['Fuel'] = df2['Fuel'].fillna(value=df2['Fuel'].mean())
    scaler = StandardScaler()
    X1 = df1.drop(columns=['speed_max', 'ndays', 'nearmiss_accel', 'nearmiss_brake', 'nearmiss_accel_under_50kmh',
                           'nearmiss_brake_above_120kmh'])
    X2 = df2[['Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
              'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight',
              'TripsinWeekdays',
              'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
              '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']]
    name1, name2 = X1.columns.tolist(), X2.columns.tolist()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.fit_transform(X2)
    result = pd.DataFrame()
    for y1 in ['nearmiss_accel', 'nearmiss_brake']:
        Y1 = df1[y1]

        poisson1 = TweedieRegressor(power=1, alpha=1, max_iter=10000).fit(X1, Y1, sample_weight=df1["ndays"])

        rfe1 = RFECV(poisson1, min_features_to_select=1, cv=KFold(10)).fit(X1, Y1)

        sfm1 = SelectFromModel(TweedieRegressor(power=1, alpha=1, max_iter=10000)).fit(X1, Y1, sample_weight=df1["ndays"])

        pi1 = permutation_importance(poisson1, X1, Y1, n_repeats=10)

        methodname = [[y1, y1, y1],['REFCV', 'SelectFromModel', 'permutation_importance']]
        result1 = pd.DataFrame(zip(rfe1.ranking_, sfm1.get_support(), map(lambda x: round(x, 3), pi1.importances_mean)), index=name1, columns=methodname)
        result = pd.concat([result, result1], axis=1)
    result.to_excel(output1)
    result_ = pd.DataFrame()
    for y2 in ['Harshacceleration', 'Harshdeceleration']:
        Y2 = df2[y2]

        poisson2 = TweedieRegressor(power=1, alpha=1, max_iter=10000).fit(X2, Y2, sample_weight=df2["Days"])

        rfe2 = RFECV(poisson1, min_features_to_select=1, cv=KFold(10)).fit(X2, Y2)

        sfm2 = SelectFromModel(TweedieRegressor(power=1, alpha=1, max_iter=10000)).fit(X2, Y2, sample_weight=df2["Days"])

        pi2 = permutation_importance(poisson2, X2, Y2, n_repeats=10)

        methodname = [[y2, y2, y2],['REFCV', 'SelectFromModel', 'permutation_importance']]
        result2 = pd.DataFrame(zip(rfe2.ranking_, sfm2.get_support(), map(lambda x: round(x, 3), pi2.importances_mean)), index=name2, columns=methodname)
        result_ = pd.concat([result_, result2], axis=1)
    result_.to_excel(output2)
