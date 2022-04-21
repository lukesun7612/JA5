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
    input1 = "./summary_Spain.xlsx"
    input2 = "./summary_China.csv"
    df1 = pd.read_excel(input1, index_col='TELEMATICSID')
    df2 = pd.read_csv(input2, index_col='ID')
    df1.columns = df1.columns.str.replace('_driver', '')
    df1.columns = df1.columns.str.replace('_media', '')
    df2['Fuel'] = df2['Fuel'].fillna(value=df2['Fuel'].mean())
    X1 = df1[['speed_mean','speed_max', 'accel_mean', 'heading_mean',
              'ndays', 'ntrips', 'ntrips_per_day', 'nweekdays', 'nweekenddays', 'ntrips_in_weekdays', 'ntrips_in_weekenddays', 'ntrips_under_15min', 'ntrips_between_15min_30min', 'ntrips_between_30min_1h', 'ntrips_between_1h_2h', 'ntrips_longer_2h',
              'distance', 'distance_max_per_day', 'distance_per_trip', 'distance_week', 'distance_weekend', 'distance_during_day', 'distance_during_night', 'distance_under_50kmh', 'distance_above_120kmh']]
    X2 = df2[['Speed', 'Brakes', 'Range', 'RPM', 'Pedal position', 'Engine fuel rate',
              'Days', 'Trips', 'TripperDay', 'TripsinDay', 'TripsinNight', 'Weekdays', 'Weekends', 'TripsinWeekdays', 'TripsinWeekends', 'Trips<15m', '15m<Trips<30m', '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h',
              'Distance', 'Fuel', 'DistanceperDays', 'DistanceperTrip', 'DistanceinWeekdays', 'DistanceinWeekends', 'DistanceinDay', 'DistanceinNight']]
    df_1 = pd.concat([df1[['nearmiss_accel', 'nearmiss_brake']], X1], 1)
    df_2 = pd.concat([df2[['Harshacceleration', 'Harshdeceleration']], X2], 1)
    dfcorr1 = X1.corr()
    dfcorr2 = X2.corr()
    ## 相关性热力图
    f1, ax1 = plt.subplots(figsize=(12, 9), dpi=100)
    ax1 = sns.heatmap(data=dfcorr2,
                center=0,
                cmap='RdBu_r',
                # cbar=False,
                xticklabels=1,
                yticklabels=1,
                annot=True,  # 图中数字文本显示
                fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
                annot_kws={'size': 8, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
                # mask=np.triu(np.ones_like(dfcorr2, dtype=np.bool)),  # 显示对脚线下面部分图
                square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
                cbar_kws={"shrink": .5},
                ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=-90, )
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, )
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    # ax1.set_xlabel('China Dataset', fontsize=8, rotation=0)
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=6)
    f2, ax2 = plt.subplots(figsize=(12, 9), dpi=100)
    ax2 = sns.heatmap(data=dfcorr1,
                center=0,
                cmap='RdBu_r',
                # cbar=False,
                xticklabels=1,
                yticklabels=1,
                annot=True,  # 图中数字文本显示
                fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
                annot_kws={'size': 10, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
                # mask=np.tril(np.ones_like(dfcorr1, dtype= np.bool)),  # 显示对脚线下面部分图
                square = True, linewidths = .5,  # 每个方格外框显示，外框宽度设置
                cbar_kws={"shrink": .5},
                ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=-90, )
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, )
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    # ax2.invert_xaxis()
    # ax2.set_xlabel('Spain Dataset', fontsize = 8, rotation=0)
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(labelsize = 6)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # ## 聚类热图
    # f3, ax3 = plt.subplots(figsize=(11, 11), dpi=100)
    # ax3 = sns.clustermap(data=dfcorr2,
    #                      center=0,
    #                      cmap='RdBu_r',
    #                      # cbar=False,
    #                      xticklabels=1,
    #                      yticklabels=1,
    #                      annot=True,  # 图中数字文本显示
    #                      fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
    #                      annot_kws={'size': 4, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
    #                      # mask=np.triu(np.ones_like(dfcorr2, dtype=np.bool)),  # 显示对脚线下面部分图
    #                      square=True,
    #                      linewidths=.5,  # 每个方格外框显示，外框宽度设置
    #                      cbar_pos=(.25, .35, .01, .5),
    #                      dendrogram_ratio=(.4, .1))
    # ax3.ax_row_dendrogram.remove()
    plt.show()
