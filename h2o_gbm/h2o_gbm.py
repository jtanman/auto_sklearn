import datetime

import feather
import h2o
import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.utils.distributions import CustomDistributionGaussian

from auto_sk.auto_sk import rmse_weighted

# # Functions to analyse and plot results
# from pandas.plotting import register_matplotlib_converters
# from matplotlib.dates import DateFormatter
# import matplotlib
# import matplotlib.pyplot as plt
# register_matplotlib_converters()

# plt.style.use('seaborn-deep')
# plt.rcParams['legend.fontsize'] = 'large'
# plt.rcParams['figure.facecolor'] = 'white'

# # evaluate number of good predictions
# def evaluate(test, predictions):
#     predictions["actual"] = test.delivery.values
#     predictions.columns = ["prediction", "actual"]
#     data_val_treated.delivery.values- predictions.prediction
#     predictions["sresidual"] = predictions.residual / np.sqrt(predictions.actual)
#     predictions["fit"] = 0
#     # if residual is positive there are not enough items in the store
#     predictions.loc[predictions.residual > 0, "fit"] = 0
#     # if residual is zero or negative there are enough or more items in the store
#     predictions.loc[predictions.residual <= 0, "fit"] = 1
#     items = predictions.shape[0]
#     more_or_perfect = sum(predictions.fit)
#     less = items - more_or_perfect
#     return (items, less, more_or_perfect)


# # print result of evaluation
# def print_evaluation(predictions, less, more_or_perfect):
#     # set if no figure on output
#     # %matplotlib inline

#     # create scatter plot to show numbers of the errors
#     name = ["fewer", "more or perfect"]
#     count = [less, more_or_perfect]
#     rank = [-4, 4]

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.axis([-10, 10, -14000, 90000])
#     ax.scatter(rank, count, s=(count * 2000), marker='o', c=["#c44e52", "#55a868"])

#     for n, c, r in zip(name, count, rank):
#         plt.annotate("{}".format(c), xy=(r, c), ha="center", va="center", color="white", weight='bold', size=15)
#         plt.annotate(
#             n,
#             xy=(r, c),
#             xytext=(0, 10),
#             textcoords="offset points",
#             ha="center",
#             va="bottom",
#             color="white",
#             weight='bold',
#             size=12,
#         )
#     plt.title("Ratio between acceptable and nonacceptable predictions", weight='bold', size=20)
#     plt.axis('off')
#     plt.show()

#     plt.figure(figsize=(20, 10))
#     n, bins, patches = plt.hist(x=predictions.sresidual, bins='auto')
#     plt.grid(axis='both')
#     plt.xlabel('Value of residual', size=20)
#     plt.ylabel('Frequency', size=20)
#     plt.title(
#         'Histogram of standardized residuals \n mean: %f\n variance: %f'
#         % (np.mean(predictions.sresidual), np.var(predictions.sresidual)),
#         weight='bold',
#         size=20,
#     )
#     maxfreq = n.max()
#     plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     for i in range(len(patches)):
#         if bins[i] > 0 and round(bins[i + 1]) >= 0:
#             patches[i].set_facecolor('#c44e52')
#         else:
#             patches[i].set_facecolor('#55a868')
#     plt.show()


# # residual analysis
# def print_residuals(predictions, predictions_custom):
#     # histograms
#     plt.figure(figsize=(20, 10))
#     n, bins, patches = plt.hist(
#         x=[predictions.sresidual, predictions_custom.sresidual], bins='auto', label=['residual', 'residual custom']
#     )
#     plt.grid(axis='both')
#     plt.xlabel('Value of standardized residual', size=22)
#     plt.ylabel('Frequency', size=20)
#     plt.title('Histograms of standardized residuals', weight='bold', size=22)
#     plt.legend(loc='upper right')
#     maxfreq = n[0].max() if n[0].max() > n[1].max() else n[1].max()
#     plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     plt.show()

#     # actual vs. predicted

#     f = plt.figure(figsize=(30, 10))
#     ax = f.add_subplot(121)
#     ax2 = f.add_subplot(122)
#     ax.scatter(data_val_treated.delivery.values, s=10, label='Gaussian', alpha=0.7)
#     ax.scatter(data_val_treated.delivery.values, s=10, label='Custom', alpha=0.7)
#     plt.grid(axis='both')
#     ax.set_xlabel('Predicted', size=20)
#     ax.set_ylabel('Actual', size=20)
#     ax.legend(loc='upper right')
#     ax.set_title("Predicted vs. Actual", weight='bold', size=22)

#     # residual error
#     ax2.scatter(predictions.prediction, predictions.sresidual, s=10, label='Gaussian', alpha=0.7)
#     ax2.scatter(predictions_custom.predict, predictions_custom.sresidual, s=10, label='Custom', alpha=0.7)
#     plt.hlines(y=0, xmin=0, xmax=200, linewidth=2)
#     plt.grid(axis='both')
#     ax2.set_xlabel('Prediction', size=20)
#     ax2.set_ylabel('Standardized residual', size=20)
#     ax2.legend(loc='upper right')
#     ax2.set_title("Standardized residual errors", weight='bold', size=22)
#     plt.show()


# # prediction analysis
# def print_predictions(predictions, predictions_custom, item, store):
#     one_item_data = predictions[(predictions.item == item) & (predictions.store == store)]
#     one_item_data_custom = predictions_custom[(predictions_custom.item == item) & (predictions_custom.store == store)]
#     fig, ax = plt.subplots(figsize=(30, 10))
#     ax.set_title(("Prediction vs. Actual item: %d store: %d") % (item, store), weight='bold', size=22)
#     ax.set_xlabel('Date', size=20)
#     ax.set_ylabel('Number of sold items', size=20)
#     plt.grid(axis='both')
#     plt.xticks(rotation=70)
#     ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
#     ax.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y"))
#     ax.xaxis.set_minor_formatter(DateFormatter("%d-%m-%Y"))
#     ax.plot_date(one_item_data.date, one_item_data.prediction, "o:", alpha=0.6, ms=10, label="predicted - Gaussian")
#     ax.plot_date(
#         one_item_data_custom.date, one_item_data_custom.prediction, "o:", ms=10, alpha=0.6, label="predicted - custom"
#     )
#     ax.plot_date(one_item_data.date, one_item_data.actual, "o", markersize=10, label="actual")
#     ax.legend(loc='upper right')


# def plot_scoring_history(history_mm, history_cmm):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_title("Training scoring history", weight='bold', size=22)
#     ax.set_xlabel('Number of trees', size=20)
#     ax.set_ylabel('Custom metric', size=20)
#     plt.grid(axis='both')
#     ax.plot(history_mm, "o:", ms=10, label="Gaussian distribution & custom metric")
#     ax.plot(history_cmm, "o:", ms=10, label="Custom distribution & custom metric")
#     ax.legend(loc='upper right')


def train_custom_gbm():

    h2o.init()

    data_train_treated = feather.read_dataframe('./data_train_treated.feather')
    data_val_treated = feather.read_dataframe('./data_val_treated.feather')

    data_train_treated = data_train_treated.sample(n=10000, axis=0)

    # X_train = data_train_treated.drop('delivery', axis=1)
    # y_train = data_train_treated['delivery']

    # X_test = data_val_treated.drop('delivery', axis=1)
    # y_test = data_val_treated['delivery']

    train_h2o = h2o.H2OFrame(data_train_treated)
    test_h2o = h2o.H2OFrame(data_val_treated)

    gbm_gaussian = H2OGradientBoostingEstimator(
        model_id="delivery_model", ntrees=50, max_depth=5, score_each_iteration=True, distribution="gaussian"
    )

    gbm_gaussian.train(y="delivery", x=train_h2o.names, training_frame=train_h2o)

    # Predict
    predictions = gbm_gaussian.predict(test_data=test_h2o).as_data_frame()

    import ipdb

    ipdb.set_trace()
    # Evalute and print summary
    # items, less, more_or_perfect = evaluate(data_val_treated, predictions)

    print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions.prediction)}')
    print(f'Proportion Late: {np.mean(data_val_treated.delivery.values > predictions.prediction)}')

    # print_evaluation(predictions, less, more_or_perfect)

    # Define asymmetric loss distribution from Gaussian distribution
    class AsymmetricLossDistribution(CustomDistributionGaussian):
        def gradient(self, y, f):
            error = (y - f) ** 2
            # more predicted items is better error than the fewer predicted items
            # if residual is positive there are not enough items in the store
            # if residual is negative or zero there are enough items in the store
            # the positive error should be set as bigger!
            return error if error < 0 else 2 * error

    # upload distribution to h2o
    name = "asymmetric"
    try:
        distribution_ref = h2o.upload_custom_distribution(
            AsymmetricLossDistribution, func_name="custom_" + name, func_file="custom_" + name + ".py"
        )
    except (IOError):
        print("This error occur in python 2.7 due to inspect package bug.")
        print(
            "You can solve this problem by saving custom distribution class to a file with .py extension and loaded it to IPython separately."
        )
        import sys

        sys.exit()

    gbm_custom = H2OGradientBoostingEstimator(
        model_id="custom_delivery_model",
        ntrees=50,
        max_depth=5,
        score_each_iteration=True,
        distribution="custom",
        custom_distribution_func=distribution_ref,
    )

    gbm_custom.train(y="delivery", x=train_h2o.names, training_frame=train_h2o)

    # Predict
    predictions_custom = gbm_custom.predict(test_data=test_h2o).as_data_frame()

    # Evalute and print summary
    print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom.predict)}')
    print(f'Proportion Late: {np.mean(data_val_treated.delivery.values > predictions_custom.predict)}')
    # items_custom, less_custom, more_or_perfect_custom = evaluate(test, predictions_custom)

    # print_evaluation(predictions_custom, less_custom, more_or_perfect_custom)

    print("original vs. custom")
    # print("actual mean:", predictions.actual.mean(), predictions_custom.actual.mean())
    print("prediction mean:", predictions.prediction.mean(), predictions_custom.predict.mean())
    print("prediction variance:", predictions.prediction.var(), predictions_custom.predict.var())
    print("residual mean:", predictions.sresidual.mean(), predictions_custom.sresidual.mean())
    print("residual variance:", predictions.sresidual.var(), predictions_custom.sresidual.var())

    # print_residuals(predictions, predictions_custom)
    # print_predictions(predictions, predictions_custom, 1, 1)
    # print_predictions(predictions, predictions_custom, 42, 2)

    # Custom asymmetric metric

    class CustomAsymmetricMseFunc:
        def map(self, pred, act, w, o, model):
            error = act[0] - pred[0]
            # more predicted items is better error than the fewer predicted items
            # if residual is positive there are not enough items in the store
            # if residual is negative or zero there are enough items in the store
            # the positive error should be set as bigger error!
            error = error if error < 0 else 2 * error
            return [error * error, 1]

        def reduce(self, l, r):
            return [l[0] + r[0], l[1] + r[1]]

        def metric(self, l):
            return np.sqrt(l[0] / l[1])

    # Upload the custom metric
    metric_ref = h2o.upload_custom_metric(CustomAsymmetricMseFunc, func_name="custom_mse", func_file="custom_mse.py")

    # Train GBM model with custom metric
    gbm_custom_mm = H2OGradientBoostingEstimator(
        model_id="custom_delivery_model_mm",
        ntrees=50,
        max_depth=5,
        score_each_iteration=True,
        stopping_metric="custom",
        stopping_tolerance=0.1,
        stopping_rounds=5,
        distribution="gaussian",
        custom_metric_func=metric_ref,
    )

    gbm_custom_mm.train(y="delivery", x=train_h2o.names, training_frame=train_h2o)

    # Predict
    predictions_custom_mm = gbm_custom_mm.predict(test_data=test_h2o).as_data_frame()

    # Evalute and print summary
    print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom_mm.predict)}')
    print(f'Proportion Late: {np.mean(data_val_treated.delivery.values > predictions_custom_mm.predict)}')
    # items_custom_mm, less_custom_mm, more_or_perfect_custom_mm = evaluate(test, predictions_custom_mm)

    # print_evaluation(predictions_custom_mm, less_custom_mm, more_or_perfect_custom_mm)

    # Train GBM model with custom metric and distribution
    gbm_custom_cmm = H2OGradientBoostingEstimator(
        model_id="custom_delivery_model_cmm",
        ntrees=50,
        max_depth=5,
        score_each_iteration=True,
        stopping_metric="custom",
        stopping_tolerance=0.1,
        stopping_rounds=5,
        distribution="custom",
        custom_metric_func=metric_ref,
        custom_distribution_func=distribution_ref,
    )

    # Predict
    predictions_custom_cmm = gbm_custom_cmm.predict(test_data=test_h2o).as_data_frame()

    # Evalute and print summary
    print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom_cmm.predict)}')
    print(f'Proportion Late: {np.mean(data_val_treated.delivery.values > predictions_custom_cmm.predict)}')
    # items_custom_cmm, less_custom_cmm, more_or_perfect_custom_cmm = evaluate(test, predictions_custom_cmm)

    # print_evaluation(predictions_custom_cmm, less_custom_cmm, more_or_perfect_custom_cmm)

    # print_residuals(predictions_custom_mm, predictions_custom_cmm)

    # plot_scoring_history(gbm_custom_mm.scoring_history()["training_custom"], gbm_custom_cmm.scoring_history()["training_custom"])

    # print_predictions(predictions, predictions_custom_cmm, 1, 1)

    # print_predictions(predictions, predictions_custom_cmm, 42, 2)
