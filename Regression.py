#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: lukes
@project: JA5
@file: Regression.py
@time: 2021/5/27 1:36
"""
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV, learning_curve
from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, mean_absolute_error, explained_variance_score, d2_tweedie_score, auc
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']
pd.set_option('display.max_columns', None)
def score_estimator(estimator, X, Y, E):
    """Score an estimator."""
    y_pred = estimator.predict(X)
    rmse = math.sqrt(mean_squared_error(Y, y_pred, sample_weight=E))
    mae = mean_absolute_error(Y, y_pred, sample_weight=E)
    # Ignore non-positive predictions, as they are invalid for the Poisson deviance.
    mask = y_pred >= 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(f"WARNING: Estimator yields invalid, non-positive predictions "
              f" for {n_masked} samples out of {n_samples}. These predictions "
              f"are ignored when computing the Poisson deviance.")
    mpd = mean_poisson_deviance(Y, y_pred, sample_weight=E)
    evs = explained_variance_score(Y, y_pred, sample_weight=E)
    d2 = d2_tweedie_score(Y, y_pred, sample_weight=E, power=1)
    return mpd, rmse, mae, evs, d2

def plot_learning_curve(estimator, X, y, names=None, score='neg_mean_poisson_deviance', color='b', cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=score,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(-train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(-test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=color,
        )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=color,
        )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color=color, label=names+"训练数据"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o--", color=color, label=names+"验证数据"
    )
    axes[0].set_title("学习过程")
    axes[0].set_xlabel("数据量")
    axes[0].set_ylabel("评估指标得分")
    axes[0].legend(loc="best")
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-", color=color, label=names)
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
        color=color,
        )
    axes[1].set_xlabel("数据量")
    axes[1].set_ylabel("模型拟合时间")
    axes[1].set_title("学习时间")
    axes[1].legend(loc="best")
    # # Plot fit_time vs score
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # axes[2].grid()
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-", color=color, label=names)
    # axes[2].fill_between(
    #     fit_time_sorted,
    #     test_scores_mean_sorted - test_scores_std_sorted,
    #     test_scores_mean_sorted + test_scores_std_sorted,
    #     alpha=0.1,
    #     color=color,
    #     )
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")
    # axes[2].legend(loc="best")
    return plt
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)
    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure)
    cumulated_claims /= cumulated_claims[-1]
    cumulated_exposure = np.cumsum(ranked_exposure)
    cumulated_exposure /= cumulated_exposure[-1]
    return cumulated_exposure, cumulated_claims


if __name__ == '__main__':
    scaler = StandardScaler()
    input1 = "./summary_China.csv"
    input2 = "./summary_Spain.xlsx"
    output = "./Table/Model contrast.xlsx"
    df1 = pd.read_csv(input1, index_col='ID')
    df2 = pd.read_excel(input2, index_col='TELEMATICSID')
    df2.columns = df2.columns.str.replace('_driver', '')
    df2.columns = df2.columns.str.replace('_media', '')
    X1 = df1[['Brakes', 'Speed', 'Range', 'RPM', 'Pedal position', 'Engine fuel rate',
                   'TripsinNight', 'TripsinWeekends',
                   'Trips<15m', '30m<Trips<1h', '1h<Trips<2h', 'Trips>2h', 'Weekdays', 'Weekends',
                   'DistanceinDay']]
    X2 = df2[['speed_mean', 'speed_max', 'accel_mean', 'heading_mean',
              'ntrips_per_day', 'ntrips_in_weekenddays', 'ntrips_under_15min', 'ntrips_between_15min_30min', 'ntrips_between_30min_1h',
              'ntrips_between_1h_2h', 'ntrips_longer_2h',
              'distance_max_per_day', 'distance_week', 'distance_weekend', 'distance_per_trip',
              'distance_during_night',  'distance_above_120kmh']]
    name1, name2 = X1.columns.tolist(), X2.columns.tolist()
    Y1 = df1[['Overspeed', 'Distance']]
    Y2 = df1[['Highspeedbrake', 'Distance']]
    Y3 = df1[['Harshacceleration', 'Distance']]
    Y4 = df1[['Harshdeceleration', 'Distance']]
    Y5 = df2[['nearmiss_accel', 'distance']]
    Y6 = df2[['nearmiss_brake', 'distance']]
    Y7 = df2[['nearmiss_accel_under_50kmh', 'distance']]
    Y8 = df2[['nearmiss_brake_above_120kmh', 'distance']]
    cv = KFold(n_splits=5)
    scores = ['neg_mean_poisson_deviance', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
    n_bins = 100
    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(16, 4))
    # # fig.subplots_adjust(bottom=0.2)
    # ax0.set_title("China Data Set")
    # _ = Y1.hist(bins=n_bins, log=False, ax=ax0)
    # ax0.set_xlabel("Harshacceleration")
    # ax1.set_title("China Data Set")
    # _ = Y2.hist(bins=n_bins, log=False, ax=ax1)
    # ax1.set_xlabel("Harshdeceleration")
    # ax2.set_title("Spain Data Set")
    # _ = Y3.hist(bins=n_bins, log=False, ax=ax2)
    # ax2.set_xlabel("nearmiss_accel")
    # ax3.set_title("Spain Data Set")
    # _ = Y4.hist(bins=n_bins, log=False, ax=ax3)
    # ax3.set_xlabel("nearmiss_brake")
    # ax0.set_ylabel("Frequency")

    result = pd.DataFrame()
    for n, (X, Y, label) in enumerate(zip([X1, X1, X1, X1, X2, X2, X2, X2], [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], ['China Data Set', 'China Data Set', 'China Data Set', 'China Data Set', 'Spain Data Set', 'Spain Data Set', 'Spain Data Set', 'Spain Data Set'])):
        result1, result2, result3, result4, result5, result6, result7 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)
        # nb = sm.NegativeBinomial(Y_train.iloc[:, 0], sm.add_constant(X_train), loglike_method='nb2', exposure=Y_train.iloc[:, 1], missing='drop').fit()  # cov_type='HC0'
        # td = sm.GLM(Y_train.iloc[:, 0], sm.add_constant(X_train), family=sm.families.Tweedie(link=sm.genmod.families.links.log, var_power=1.6), exposure=Y_train.iloc[:, 1], missing='drop').fit()
        poisson = PoissonRegressor(max_iter=1000, warm_start=True).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        # tweedie = TweedieRegressor(power=1.1, link='log', max_iter=1000).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        # tw = GridSearchCV(estimator=tweedie, param_grid={'power': np.arange(1, 2, 0.01)}, scoring=scores, cv=cv, refit='neg_mean_poisson_deviance', return_train_score=True).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        # print(tw.best_params_)
        # print(tw.best_score_)
        # tw_result = tw.cv_results_
        # # Plot best power
        # plt.figure(figsize=(13, 13))
        # plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
        # plt.xlabel("Power")
        # plt.ylabel("Score")
        # ax = plt.gca()
        # # Get the regular numpy array from the MaskedArray
        # X_axis = np.array(tw_result["param_power"].data, dtype=float)
        # for scorer, color in zip(['neg_mean_poisson_deviance'], ["g"]):
        #     for sample, style in (("train", "--"), ("test", "-")):
        #         sample_score_mean = -tw_result["mean_%s_%s" % (sample, scorer)]
        #         sample_score_std = tw_result["std_%s_%s" % (sample, scorer)]
        #         ax.fill_between(
        #             X_axis,
        #             sample_score_mean-sample_score_std,
        #             sample_score_mean+sample_score_std,
        #             alpha=0.1 if sample == "test" else 0,
        #             color=color,
        #             )
        #         ax.plot(
        #             X_axis,
        #             sample_score_mean,
        #             style,
        #             color=color,
        #             alpha=1 if sample == "test" else 0.7,
        #             label="%s (%s)" % (scorer, sample),
        #             )
        #         ax.set_ylim(min(sample_score_mean-sample_score_std), max(sample_score_mean+sample_score_std))
        #     best_index = np.nonzero(tw_result["rank_test_%s" % scorer] == 1)[0][0]
        #     best_score = -tw_result["mean_test_%s" % scorer][best_index]
        #     # Plot a dotted vertical line at the best score for that scorer marked by x
        #     ax.plot(
        #         [
        #             X_axis[best_index],
        #         ]*2
        #         ,
        #         [0,
        #          best_score],
        #         linestyle="-.",
        #         color=color,
        #         marker="x",
        #         markeredgewidth=3,
        #         ms=8,
        #         )
        #     # Annotate the best score for that scorer
        #     ax.annotate("%.3f" % best_score, (X_axis[best_index], best_score+0.5))
        #     ax.annotate("%.2f" % X_axis[best_index], (X_axis[best_index], min(sample_score_mean-sample_score_std)+0.5))
        #
        # ax.set_xlim(1, 2)
        # plt.legend(loc="best")
        # plt.grid(False)

        ## Negative binomial
        # result7['Data Set'] = [label]
        # result7['Obsevations'] = Y_train.iloc[:, 0].name
        # result7['Estimator'] = ['NB']
        # result7['train MPD'], result7['train RMSE'], result7['train MAE'], result7['train EVS'], result7['train D2'] = score_estimator(poisson, X_train, Y_train.iloc[:, 0], Y_train.iloc[:, 1])
        # result7['test MPD'], result7['test RMSE'], result7['test MAE'], result7['test EVS'], result7['test D2'] = score_estimator(poisson, X_test, Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        # result = pd.concat([result, result7])
        # Random Forest
        regressor1 = RandomForestRegressor(criterion="poisson", random_state=n, n_jobs=4, warm_start=True).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        result1['Data Set'] = [label]
        result1['Obsevations'] = Y_train.iloc[:, 0].name
        result1['Estimator'] = ['RF']
        result1['train MPD'], result1['train RMSE'], result1['train MAE'], result1['train EVS'], result1['train D2'] = score_estimator(regressor1, X_train, Y_train.iloc[:, 0], Y_train.iloc[:, 1])
        result1['test MPD'], result1['test RMSE'], result1['test MAE'], result1['test EVS'], result1['test D2'] = score_estimator(regressor1, X_test, Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        result = pd.concat([result, result1])
        # Bagging-Negative binomial
        regressor2 = BaggingRegressor(base_estimator=PoissonRegressor(max_iter=1000, warm_start=True), n_estimators=100, random_state=n, n_jobs=4, warm_start=True).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        result2['Data Set'] = [label]
        result2['Obsevations'] = Y_train.iloc[:, 0].name
        result2['Estimator'] = ['Bagging-Poisson']
        result2['train MPD'], result2['train RMSE'], result2['train MAE'], result2['train EVS'], result2['train D2'] = score_estimator(regressor2, X_train, Y_train.iloc[:, 0], Y_train.iloc[:, 1])
        result2['test MPD'], result2['test RMSE'], result2['test MAE'], result2['test EVS'], result2['test D2'] = score_estimator(regressor2, X_test, Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        result = pd.concat([result, result2])
        # HGBR evaluation
        regressor3 = HistGradientBoostingRegressor(loss='poisson', max_iter=300, random_state=n).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        result3['Data Set'] = [label]
        result3['Obsevations'] = Y_train.iloc[:, 0].name
        result3['Estimator'] = ['HGBRT']
        result3['train MPD'], result3['train RMSE'], result3['train MAE'], result3['train EVS'], result3['train D2'] = score_estimator(regressor3, X_train, Y_train.iloc[:, 0], Y_train.iloc[:, 1])
        result3['test MPD'], result3['test RMSE'], result3['test MAE'], result3['test EVS'], result3['test D2'] = score_estimator(regressor3, X_test, Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        result = pd.concat([result, result3])
        # Adaboost-Negative binomial
        regressor4 = AdaBoostRegressor(base_estimator=PoissonRegressor(max_iter=1000, warm_start=True), n_estimators=100, loss='exponential', random_state=n).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        result4['Data Set'] = [label]
        result4['Obsevations'] = Y_train.iloc[:, 0].name
        result4['Estimator'] = ['Adaboost-Poisson']
        result4['train MPD'], result4['train RMSE'], result4['train MAE'], result4['train EVS'], result4['train D2'] = score_estimator(regressor4, X_train, Y_train.iloc[:, 0], Y_train.iloc[:, 1])
        result4['test MPD'], result4['test RMSE'], result4['test MAE'], result4['test EVS'], result4['test D2'] = score_estimator(regressor4, X_test, Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        result = pd.concat([result, result4])
        # Voting
        estimator = [('RF', regressor1), ('Bagging', regressor2), ('HGBRT', regressor3), ('Adaboost', regressor4)]
        regressor5 = VotingRegressor(estimators=estimator).fit(X_train, Y_train.iloc[:, 0], sample_weight=Y_train.iloc[:, 1])
        result5['Data Set'] = [label]
        result5['Obsevations'] = Y_train.iloc[:, 0].name
        result5['Estimator'] = ['Voting']
        result5['train MPD'], result5['train RMSE'], result5['train MAE'], result5['train EVS'], result5['train D2'] = score_estimator(regressor5, X_train, Y_train.iloc[:, 0], Y_train.iloc[:, 1])
        result5['test MPD'], result5['test RMSE'], result5['test MAE'], result5['test EVS'], result5['test D2'] = score_estimator(regressor5, X_test, Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        result = pd.concat([result, result5])

        # # Plot learning curves
        # _, axes = plt.subplots(1, 2, figsize=(40, 20))
        # plot_learning_curve(estimator=regressor1, X=X_train, y=Y_train.iloc[:, 0], names='RF', color='g', cv=cv, n_jobs=4)
        # plot_learning_curve(estimator=regressor2, X=X_train, y=Y_train.iloc[:, 0], names='Bagging-Poisson', color='y', cv=cv, n_jobs=4)
        # plot_learning_curve(estimator=regressor3, X=X_train, y=Y_train.iloc[:, 0], names='HGBRT', color='b', cv=cv, n_jobs=4)
        # plot_learning_curve(estimator=regressor4, X=X_train, y=Y_train.iloc[:, 0], names='Adaboost-Poisson', color='c', cv=cv, n_jobs=4)
        # plot_learning_curve(estimator=regressor5, X=X_train, y=Y_train.iloc[:, 0], names='Voting', color='r', cv=cv, n_jobs=4)

        # Plot lorenz curves
        fig, ax = plt.subplots(figsize=(20, 20))
        for model, name, color in zip([regressor1, regressor2, regressor3, regressor4, regressor5], ['RF', 'Bagging-Poisson', 'HGBRT', 'Adaboost-Poisson', 'Voting'], ['g', 'y', 'b', 'c', 'r']):
            y_pred = model.predict(X_test)
            cum_exposure, cum_claims = lorenz_curve(Y_test.iloc[:, 0], y_pred, Y_test.iloc[:, 1])
            gini = 1 - 2 * auc(cum_exposure, cum_claims)
            label = "{}洛伦兹曲线 (Gini: {:.3f})".format(name, gini)
            ax.plot(cum_exposure, cum_claims, linestyle="-", color=color, label=label)
        # Original: y_pred == y_test
        cum_exposure, cum_claims = lorenz_curve(Y_test.iloc[:, 0], Y_test.iloc[:, 0], Y_test.iloc[:, 1])
        gini = 1 - 2 * auc(cum_exposure, cum_claims)
        label = "实际值洛伦兹曲线 (Gini: {:.3f})".format(gini)
        ax.plot(cum_exposure, cum_claims, linestyle="-.", color="gray", label=label)
        # Random Baseline
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="绝对平等线")
        ax.set(
        title="高风险驾驶事件分布情况",
        xlabel="行驶总里程的累计百分比（从高风险驾驶事件发生最少到最多）",
        ylabel="高风险驾驶事件的累计百分比",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")

        # plot frequency histgram
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 80), sharey=True)
        # fig.subplots_adjust(bottom=0.2)
        y_pred0 = Y_test.iloc[:, 0]
        pd.Series(y_pred0).hist(bins=n_bins, log=False, ax=axes[0, 0], color='gray')
        axes[0,0].set(
            title='实际值',
            # yscale='log',
            xlabel=Y_train.iloc[:, 0].name,
            ylabel="频率",

        )
        y_pred1 = regressor1.predict(X_test)
        pd.Series(y_pred1).hist(bins=n_bins, log=False, ax=axes[0,1], color='g')
        axes[0,1].set(
            title='RF',
            # yscale='log',
            xlabel=Y_train.iloc[:, 0].name,

        )
        y_pred2 = regressor2.predict(X_test)
        pd.Series(y_pred2).hist(bins=n_bins, log=False, ax=axes[1,0], color='y')
        axes[1,0].set(
            title='Bagging-Poisson',
            # yscale='log',
            xlabel=Y_train.iloc[:, 0].name,
            ylabel="频率",

        )
        y_pred3 = regressor3.predict(X_test)
        pd.Series(y_pred3).hist(bins=n_bins, log=False, ax=axes[1,1], color='b')
        axes[1,1].set(
            title='HGBRT',
            # yscale='log',
            xlabel=Y_train.iloc[:, 0].name,

        )
        y_pred4 = regressor4.predict(X_test)
        pd.Series(y_pred4).hist(bins=n_bins, log=False, ax=axes[2,0], color='c')
        axes[2,0].set(
            title='AdaBoost-Poisson',
            # yscale='log',
            xlabel=Y_train.iloc[:, 0].name,
            ylabel="频率",

        )
        y_pred5 = regressor5.predict(X_test)
        pd.Series(y_pred5).hist(bins=n_bins, log=False, ax=axes[2,1], color='r')
        axes[2,1].set(
            title='Voting',
            # yscale='log',
            xlabel=Y_train.iloc[:, 0].name,

        )
        plt.tight_layout()
    print(result)
    plt.show()
    result.to_excel(output, index=False)