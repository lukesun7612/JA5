#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA5
@file: Heatmap.py
@time: 2021/5/19 17:51
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





if __name__ == '__main__':
    input1 = "./summary_drivers.xlsx"
    input2 = "./summary0.csv"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    df2 = df2.fillna(0)
    X1 = df1.drop(columns=['speed_max','nearmiss_accel','nearmiss_brake','nearmiss_accel_under_50kmh','nearmiss_brake_above_120kmh'])
    X2 = df2[[  'Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
               'Days', 'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight', 'TripsinWeekdays',
               'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
               '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h']]
    df_1 = pd.concat([df1[['nearmiss_accel', 'nearmiss_brake']], X1], 1)
    df_2 = pd.concat([df2[['Harshacceleration', 'Harshdeceleration']], X2], 1)
    dfcorr1 = df_1.corr()
    dfcorr2 = df_2.corr()
    ## 相关性热力图
    f1, ax1 = plt.subplots(figsize=(12, 9), dpi=100)
    ax1 = sns.heatmap(data=dfcorr2,
                center=0,
                cmap='RdBu_r',
                # cbar=False,
                xticklabels=4,
                yticklabels=1,
                annot=True,  # 图中数字文本显示
                fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
                annot_kws={'size': 4, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
                # mask=np.triu(np.ones_like(dfcorr2, dtype=np.bool)),  # 显示对脚线下面部分图
                square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
                cbar_kws={"shrink": .5},
                ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, )
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, )
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlabel('China', fontsize=8, rotation=0)
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=4.3)
    f2, ax2 = plt.subplots(figsize=(12, 9), dpi=100)
    ax2 = sns.heatmap(data=dfcorr1,
                center=0,
                cmap='RdBu_r',
                # cbar=False,
                xticklabels=6,
                yticklabels=1,
                annot=True,  # 图中数字文本显示
                fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
                annot_kws={'size': 4, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
                # mask=np.tril(np.ones_like(dfcorr1, dtype= np.bool)),  # 显示对脚线下面部分图
                square = True, linewidths = .5,  # 每个方格外框显示，外框宽度设置
                cbar_kws={"shrink": .5},
                ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, )
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, )
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    # ax2.invert_xaxis()
    ax2.set_xlabel('Spain', fontsize = 8, rotation=0)
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(labelsize = 4.3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # ## 聚类热图
    # f3, ax3 = plt.subplots(figsize=(11, 11), dpi=100)
    # ax3 = sns.clustermap(data=dfcorr2,
    #                      center=0,
    #                      cmap='RdBu_r',
    #                      # cbar=False,
    #                      xticklabels=4,
    #                      yticklabels=1,
    #                      annot=True,  # 图中数字文本显示
    #                      fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
    #                      annot_kws={'size': 4, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
    #                      # mask=np.triu(np.ones_like(dfcorr2, dtype=np.bool)),  # 显示对脚线下面部分图
    #                      square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
    #                      cbar_kws={"shrink": .5})

    plt.show()