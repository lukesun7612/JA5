#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA4
@file: spaindatadescription.py
@time: 2021/3/3 19:08
"""
# import missingno as msno
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
input1 = "./summary_Spain.xlsx"
input2 = "./summary_China.csv"
df1 = pd.read_excel(input1, index_col='TELEMATICSID',)
df2 = pd.read_csv(input2, index_col='ID')
df1.columns = df1.columns.str.replace('_driver', '')
df1.columns = df1.columns.str.replace('_media', '')
df2['Fuel'] = df2['Fuel'].fillna(value=df2['Fuel'].mean())

# msno.bar(data)

df1.describe().T.to_csv('./datadescribe_Spain.csv', mode='w')
df2.describe().T.to_csv('./datadescribe_China.csv', mode='w')

if __name__ == '__main__':
    pass
