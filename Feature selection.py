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

def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


if __name__ == '__main__':
    input1 = "./summary_drivers.xlsx"
    input2 = "./summary0.csv"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    df2 = df2.fillna(0)
    scaler = StandardScaler()
    X1 = df1.drop(columns=['speed_max', 'nearmiss_accel', 'nearmiss_brake', 'nearmiss_accel_under_50kmh',
                           'nearmiss_brake_above_120kmh'])
    X2 = df2[['Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
              'Days', 'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight',
              'TripsinWeekdays',
              'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
              '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']]
    df_1 = pd.concat([df1[['nearmiss_accel', 'nearmiss_brake']], X1], 1)
    df_2 = pd.concat([df2[['Harshacceleration', 'Harshdeceleration']], X2], 1)
    X1 = scaler.fit_transform(X1)
    X2 = scaler.fit_transform(X2)
    Y1 = df1['nearmiss_accel']
    Y2 = df2['Harshacceleration']
    names = ['Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
             'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight',
             'TripsinWeekdays',
             'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
             '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']
    nb = TweedieRegressor(power=1, alpha=1, max_iter=10000)
    nb.fit(X1, Y1, sample_weight=df1["ndays"])
    rfe = RFECV(nb, min_features_to_select=1, cv=KFold(5))
    rfe.fit(X1, Y1)
    sfm = SelectFromModel(TweedieRegressor(power=1, alpha=1, max_iter=10000)).fit(X1, Y1, sample_weight=df1["ndays"])
    lasso = LassoCV(cv=5, random_state=0).fit(X1, Y1 / df1["ndays"])
    print("Features sorted by their rank:")
    # print(sorted(zip(map(lambda x: round(x, 3), rfe.ranking_), names)))
    # df = pd.DataFrame(rfe.ranking_, index=names).T
    print(rfe.ranking_)
    print(lasso.coef_)