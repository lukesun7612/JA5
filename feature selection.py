#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA5
@file: feature selection.py
@time: 2021/5/19 17:51
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import TweedieRegressor, LassoCV
from sklearn.model_selection import KFold


def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


if __name__ == '__main__':
    input1 = "./summary_drivers.xlsx"
    input2 = "./summary0.csv"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID')
    df2 = pd.read_csv(input2, index_col='ID')
    df2 = df2.fillna(0)
    scaler = StandardScaler()
    X = df2[[  'Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
               'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight', 'TripsinWeekdays',
               'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
               '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']]
    X = scaler.fit_transform(X)
    Y = df2['Harshacceleration']
    names = [  'Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
               'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight', 'TripsinWeekdays',
               'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
               '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']
    nb = TweedieRegressor(power=1.8, alpha=.1, max_iter=10000)
    nb.fit(X, Y, sample_weight=df2["Days"])
    rfe = RFECV(nb, min_features_to_select=1, cv=KFold())
    rfe.fit(X, Y)
    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x: round(x, 5), rfe.ranking_), names)))
    df = pd.DataFrame(rfe.ranking_, index=names).T

    print(rfe.support_)