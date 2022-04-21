#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: data description
@file: creatsummary.py
@time: 2020/4/9 15:26
"""

import pandas as pd
import numpy as np
import os
from interval import Interval

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def fun(x):
    if x > 1:
        return 0
    else:
        return x


def diff_value(df_column_name):
    a = list(np.array(df_column_name))
    if len(a) == 0:
        return 0
    else:
        return a[-1] - a[0]


def Range(longitude, latitude):
    a = 10 ** (-6) * np.array([longitude.min(), latitude.min()])
    b = 10 ** (-6) * np.array([longitude.max(), latitude.max()])
    dist = np.linalg.norm(b - a)
    return dist


def avgBrake(dataframe):
    if dataframe['Brake times'].sum() > 3 * diff_value(dataframe['Integral kilometer']):
        n = np.rint(np.mean([dataframe['Brake switch'].sum(), dataframe['Brake times'].sum()]))
    else:
        n = dataframe['Brake times'].sum()
    return n


def highspeedbrake(df):
    df['highspeedbrake'] = np.where((df['Speed'] > 90) & ((df['Brake times'] > 0) | (df['Brake switch'] > 0)), 1, 0)
    return df['highspeedbrake'].sum()


def hashaccelerate(df, up=0.556):
    df['hashaccelerate'] = np.where(df['accelerated speed'] > up, 1, 0)
    return df['hashaccelerate'].sum()


def hashdecelerate(df, low=-0.556):
    df['hashdecelerate'] = np.where(df['accelerated speed'] < low, 1, 0)
    return df['hashdecelerate'].sum()


def overspeed(df):
    df['overspeed'] = np.where(df['Speed'] > 99, 1, 0)
    return df['overspeed'].sum()


def getAbj(se):
    '''
    得到相邻数据的里程差和时间差信息
    :param se: 里程信息，以Series的形式
    :return: 行程差信息列表和时间差信息列表
    '''
    mile_dist = se[1:].values - se[:-1].values
    mile_dist_time = (se[1:].index.values - se[:-1].index.values) / np.timedelta64(1, 's')
    mile_dist = pd.Series(mile_dist)
    mile_dist_time = pd.Series(mile_dist_time)
    return mile_dist, mile_dist_time


def split_journey(df, rotate=100, dura=5):
    '''
    切分行程
    :param df: 输入信息，这里是dataframe的形式，信息包括'发动机转速'
    :param rotate: 转速阈值，超过阈值即认为在行程中
    :param dura: 转速不大于0超过dura即认为行程结束
    :return: 两部分：行程切分点列表和行程时间长度列表
    '''
    df['GPS time'] = df['GPS time'].astype('datetime64')
    df = df.set_index('GPS time', drop=False)
    df['trip'] = np.where(df['RPM'] >= rotate, 1, 0)  # 将转速超过rotate的部分标记为行程内
    biaoji, mile_dist_time1 = getAbj(df['trip'])

    if df['trip'][0] == 1:
        biaoji[0] = 1
    else:
        biaoji[0] = -1

    if df['trip'][-1] == 1:
        biaoji[-1] = -1
    else:
        biaoji[-1] = 1

    biaoji = biaoji[biaoji != 0]
    biao_idx = biaoji.index  # 1到-1时run

    segment, segment_time = [], []
    seg_start, seg_end, tripday, daytripcount, tripperday, nighttripcount, weekdaycount, weekendcount, \
    trip15, trip30, trip60, trip120, trip120m, daytripdist, nighttripdist, \
    weekdaydist, weekenddist = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    weekend1 = Interval('2018-06-30', '2018-07-01')
    weekend2 = Interval('2018-07-07', '2018-07-08')
    day = Interval('05:00:00', '20:00:00')
    assign = True
    for s, e in zip(biao_idx[:-1], biao_idx[1:]):
        start = 0 if s == 0 else s + 1
        duration = mile_dist_time1[start: e + 1].sum()
        assert s != e
        if biaoji[s] == 1 and assign:
            seg_start = start
            assign = False
        elif biaoji[s] == -1:
            if duration > dura * 60 and s != 0:
                seg_end = start
                seg_time = mile_dist_time1[seg_start: seg_end].sum() / 60
                segment.append((seg_start, seg_end))
                segment_time.append(seg_time)
                if df['GPS time'][seg_start].strftime("%H:%M:%S") in day and df['GPS time'][seg_end].strftime(
                        "%H:%M:%S") in day:
                    daytripcount += 1
                    daytripdist += diff_value(df['3-GPS kilometer(km)'][seg_start:seg_end])
                elif df['GPS time'][seg_start].strftime("%H:%M:%S") not in day or df['GPS time'][seg_end].strftime(
                        "%H:%M:%S") not in day:
                    nighttripcount += 1
                    nighttripdist += diff_value(df['3-GPS kilometer(km)'][seg_start:seg_end])
                if (df['GPS time'][seg_start].strftime("%Y-%m-%d") in weekend1) or (
                        df['GPS time'][seg_start].strftime("%Y-%m-%d") in weekend2) or (
                        df['GPS time'][seg_end].strftime("%Y-%m-%d") in weekend1) or (
                        df['GPS time'][seg_end].strftime("%Y-%m-%d") in weekend2):
                    weekendcount += 1
                    weekenddist += diff_value(df['3-GPS kilometer(km)'][seg_start:seg_end])
                elif (df['GPS time'][seg_start].strftime("%Y-%m-%d") not in weekend1) and (
                        df['GPS time'][seg_start].strftime("%Y-%m-%d") not in weekend2) and (
                        df['GPS time'][seg_end].strftime("%Y-%m-%d") not in weekend1) and (
                        df['GPS time'][seg_end].strftime("%Y-%m-%d") not in weekend2):
                    weekdaydist += diff_value(df['3-GPS kilometer(km)'][seg_start:seg_end])
                if seg_time < 15:
                    trip15 += 1
                elif 15 <= seg_time < 30:
                    trip30 += 1
                elif 30 <= seg_time < 60:
                    trip60 += 1
                elif 60 <= seg_time < 120:
                    trip120 += 1
                elif seg_time >= 120:
                    trip120m += 1
                assign = True
    if len(segment) == 0:
        tripday = 0
    elif len(segment) > 0:
        d = (df['GPS time'][segment[-1][1]] - df['GPS time'][segment[0][0]])
        tripday = pd.to_timedelta([d]).astype('timedelta64[D]')[0]+1
    tripcount = len(segment)
    weekdaycount = tripcount - weekendcount
    nighttripcount = tripcount - daytripcount
    tripperday = (tripcount / tripday) if tripday > 0 else 0
    return tripday, tripcount, tripperday, daytripcount, daytripdist, nighttripcount, nighttripdist, weekdaycount, weekdaydist, weekendcount, weekenddist, trip15, trip30, trip60, trip120, trip120m


if __name__ == '__main__':
    input = "D:/result/dataset0"
    output = "E:/博士/论文/JA5/summary0.csv"
    result = pd.DataFrame()
    count = 0
    columns = ['ID', 'Overspeed', 'Highspeedbrake', 'Harshacceleration', 'Harshdeceleration',
               'Distance', 'Fuel', 'Brakes', 'Speed', 'Range', 'RPM', 'Accelerator pedal position', 'Engine fuel rate',
               'Days', 'Trips', 'TripperDays', 'TripsinDay', 'DistanceinDay', 'TripsinNight', 'DistanceinNight', 'TripsinWeekdays',
               'DistanceinWeekdays', 'TripsinWeekends', 'DistanceinWeekends', 'Trips<15m', '15m<Trips<30m',
               '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h', 'DistanceperDays']
    for i, file in enumerate(os.listdir(input)):
        print(i, file)
        filepath = os.path.join(input, file)
        df = pd.read_csv(filepath, header=0)
        df = df.drop_duplicates(['GPS time'])
        df = df.rename(columns={'Selected speed(km/h)': 'Speed'})
        df = df.loc[df['Longitude'].apply(lambda x: x > 0)].loc[df['Latitude'].apply(lambda y: y > 0)]
        df['Brake switch'] = df['Brake switch'].apply(lambda x: fun(x))
        df['speeddiff'] = df['Speed'].diff(1) / 3.6
        df['timediff'] = df['GPS time'].astype('datetime64').diff().astype('timedelta64[s]')
        df['accelerated speed'] = df['speeddiff'] / df['timediff']
        record = -np.ones([29], dtype=np.float64)
        record[0] = file[:11]
        record[1] = overspeed(df)
        record[2] = highspeedbrake(df)
        record[3] = hashaccelerate(df)
        record[4] = hashdecelerate(df)
        record[6] = diff_value(df['Integral fuel consumption'])
        record[7] = avgBrake(df)
        record[8] = df['Speed'].fillna(0).mean()
        record[9] = Range(df['Longitude'], df['Latitude'])
        record[10] = df['RPM'].fillna(0).mean()
        record[11] = df['Accelerator pedal position'].fillna(0).mean()
        record[12] = df['Engine fuel rate'].fillna(0).mean()
        record[13], record[14], record[15], record[16], record[17], record[18], record[19], record[20], record[21], \
        record[22], record[23], record[24], record[25], record[26], record[27], record[28] = split_journey(df)
        record[5] = record[17] + record[19]
        record[29] = record[5]/record[13]
        res = pd.DataFrame(np.array(record.tolist()).reshape(1, 30), columns=columns)
        if record[5] < 2:
            pass
        else:
            result = pd.concat([result, res])
            count += 1
        print(count)
    result = result.set_index('ID')
    result.to_csv(output, mode='w')
