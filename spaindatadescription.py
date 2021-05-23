#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA4
@file: spaindatadescription.py
@time: 2021/3/3 19:08
"""
import missingno as msno
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
data = pd.read_excel('./summary_drivers.xlsx')

msno.bar(data)

data.describe().T.to_csv('./datadescribe.csv',mode='w')


if __name__ == '__main__':
    pass
