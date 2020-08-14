import datetime
import time
import os
import itertools

import feather
import h2o
import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.utils.distributions import CustomDistributionGaussian
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

from auto_sk.auto_sk import rmse_weighted


h2o.init()

def grid_to_summary(grid):

    table = []
    for model in grid.models:
        model_summary = model._model_json["output"]["model_summary"]
        r_values = list(model_summary.cell_values[0])
        r_values[0] = model.model_id

        model_metrics = model._model_json['output']['validation_metrics']
        if model_metrics._metric_json==None:
            print("WARNING: Model metrics cannot be calculated and metric_json is empty due to the absence of the response column in your dataset.")
            return
        metric_type = model_metrics._metric_json['__meta']['schema_type']
        types_w_glm = ['ModelMetricsRegressionGLM', 'ModelMetricsRegressionGLMGeneric', 'ModelMetricsBinomialGLM',
                       'ModelMetricsBinomialGLMGeneric', 'ModelMetricsHGLMGaussianGaussian', 
                       'ModelMetricsHGLMGaussianGaussianGeneric']
        types_w_clustering = ['ModelMetricsClustering']
        types_w_mult = ['ModelMetricsMultinomial', 'ModelMetricsMultinomialGeneric']
        types_w_ord = ['ModelMetricsOrdinal', 'ModelMetricsOrdinalGeneric']
        types_w_bin = ['ModelMetricsBinomial', 'ModelMetricsBinomialGeneric', 'ModelMetricsBinomialGLM', 'ModelMetricsBinomialGLMGeneric']
        types_w_r2 = ['ModelMetricsRegressionGLM', 'ModelMetricsRegressionGLMGeneric']
        types_w_mean_residual_deviance = ['ModelMetricsRegressionGLM', 'ModelMetricsRegressionGLMGeneric',
                                          'ModelMetricsRegression', 'ModelMetricsRegressionGeneric']
        types_w_mean_absolute_error = ['ModelMetricsRegressionGLM', 'ModelMetricsRegressionGLMGeneric',
                                       'ModelMetricsRegression', 'ModelMetricsRegressionGeneric']
        types_w_logloss = types_w_bin + types_w_mult+types_w_ord
        types_w_dim = ["ModelMetricsGLRM"]
        types_w_anomaly = ['ModelMetricsAnomaly']
        
        if metric_type not in types_w_anomaly:
            r_values.append(model_metrics.mse())
            r_values.append(model_metrics.rmse())
        if metric_type in types_w_mean_absolute_error:
            r_values.append(model_metrics.mae())
            r_values.append(model_metrics.rmsle())
        if metric_type in types_w_r2:
            r_values.append(model_metrics.r2())
        if metric_type in types_w_mean_residual_deviance:
            r_values.append(model_metrics.mean_residual_deviance())
        if model_metrics.custom_metric_name():
            r_values.append(model_metrics.custom_metric_value())

        table.append(r_values)

    keys = ['model_ids'] + model_summary.col_header[1:]
    if metric_type not in types_w_anomaly:
        keys.extend(['mse', 'rmse'])
    if metric_type in types_w_mean_absolute_error:
        keys.extend(['mae', 'rmsle'])
    if metric_type in types_w_r2:
        keys.append('r2')
    if metric_type in types_w_mean_residual_deviance:
        keys.append('mean_residual_deviance')
    if model_metrics.custom_metric_name():
        keys.append(model_metrics.custom_metric_name())

    df = pd.DataFrame(table, columns=keys)
    
    return(df)


def grid_to_params(grid):
    """Print models sorted by metric.

    :examples:

    >>> from h2o.estimators import H2ODeepLearningEstimator
    >>> from h2o.grid.grid_search import H2OGridSearch
    >>> insurance = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/glm_test/insurance.csv")
    >>> insurance["offset"] = insurance["Holders"].log()
    >>> insurance["Group"] = insurance["Group"].asfactor()
    >>> insurance["Age"] = insurance["Age"].asfactor()
    >>> insurance["District"] = insurance["District"].asfactor()
    >>> hyper_params = {'huber_alpha': [0.2,0.5],
    ...                 'quantile_alpha': [0.2,0.6]}
    >>> from h2o.estimators import H2ODeepLearningEstimator
    >>> gs = H2OGridSearch(H2ODeepLearningEstimator(epochs=5),
    ...                    hyper_params)
    >>> gs.train(x=list(range(3)),y="Claims", training_frame=insurance)
    >>> gs.show()
    """
    hyper_combos = itertools.product(*list(grid.hyper_params.values()))
    if not grid.models:
        c_values = [[idx + 1, list(val)] for idx, val in enumerate(hyper_combos)]
        print(H2OTwoDimTable(
            col_header=['Model', 'Hyperparameters: [' + ', '.join(list(grid.hyper_params.keys())) + ']'],
            table_header='Grid Search of Model ' + grid.model.__class__.__name__, cell_values=c_values))
    else:
        return(pd.DataFrame(grid.sorted_metric_table()))

def grid_to_df(grid, sort_by=None, ascending=True):

    df_params = grid_to_show(grid)
    df_summary = grid_to_summary(grid)

    df_ret = pd.concat([df_params, df_summary], axis=1)
    df_ret = df_ret.loc[:,~df_ret.columns.duplicated()]
    if sort_by is not None:
        df_ret.sort_values(sort_by, ascending)
    return df_ret


# evaluate number of good predictions
def evaluate(test, predictions):

    predictions["actual"] = test.delivery.values
    predictions.columns = ["predict", "actual"]
    predictions["residual"] = predictions.actual - predictions.predict
    predictions["sresidual"] = predictions.residual / np.sqrt(predictions.actual)
    predictions["fit"] = 0
    # if residual is positive there are not enough items in the store
    predictions.loc[predictions.residual > 0, "fit"] = 0
    # if residual is zero or negative there are enough or more items in the store
    predictions.loc[predictions.residual <= 0, "fit"] = 1
    deliveries = predictions.shape[0]
    early = sum(predictions.fit)
    late = deliveries - early
    return (deliveries, late, early)


data_train_treated = feather.read_dataframe('./data_train_treated.feather')
data_val_treated = feather.read_dataframe('./data_val_treated.feather')
data_pred_treated = feather.read_dataframe('./data_pred_treated.feather')

# data_train_treated = data_train_treated.sample(n=10000, axis=0)

ind_vars = list(data_train_treated.columns)
ind_vars.remove('delivery')

train_h2o = h2o.H2OFrame(data_train_treated)
test_h2o = h2o.H2OFrame(data_val_treated)
pred_h2o = h2o.H2OFrame(data_pred_treated)

# gbm_gaussian = H2OGradientBoostingEstimator(
#     model_id="delivery_model", ntrees=50, max_depth=5, score_each_iteration=True, distribution="gaussian"
# )

# gbm_gaussian.train(y="delivery", x=ind_vars, training_frame=train_h2o)

# # Predict
# predictions = gbm_gaussian.predict(test_data=test_h2o).as_data_frame()
# deliveries, late, early = evaluate(data_val_treated, predictions)

# print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
# print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions.predict)}')

# Define asymmetric loss distribution from Gaussian distribution
class AsymmetricLossDistribution(CustomDistributionGaussian):
    def gradient(self, y, f):
        error = y - f
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

# gbm_custom = H2OGradientBoostingEstimator(
#     model_id="custom_delivery_model",
#     ntrees=50,
#     max_depth=5,
#     score_each_iteration=True,
#     distribution="custom",
#     custom_distribution_func=distribution_ref,
# )

# gbm_custom.train(y="delivery", x=train_h2o.names, training_frame=train_h2o)

# # Predict
# predictions_custom = gbm_custom.predict(test_data=test_h2o).as_data_frame()
# deliveries, late, early = evaluate(data_val_treated, predictions_custom)

# # Evalute and print summary
# print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
# print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom.predict)}')

# print("original vs. custom")
# print("prediction mean:", predictions.predict.mean(), predictions_custom.predict.mean())
# print("prediction variance:", predictions.predict.var(), predictions_custom.predict.var())
# print("residual mean:", predictions.sresidual.mean(), predictions_custom.sresidual.mean())
# print("residual variance:", predictions.sresidual.var(), predictions_custom.sresidual.var())

# Custom asymmetric metric

# class CustomAsymmetricMseFunc:
#     def map(self, pred, act, w, o, model):
#         error = act[0] - pred[0]
#         error = error if error < 0 else 2 * error
#         return [error * error, 1]

#     def reduce(self, l, r):
#         return [l[0] + r[0], l[1] + r[1]]

#     def metric(self, l):
#         import java.lang.Math as math
#         return np.sqrt(l[0] / l[1])

# # Upload the custom metric
# metric_ref = h2o.upload_custom_metric(CustomAsymmetricMseFunc, func_name="custom_mse", func_file="custom_mse.py")


class CustomRmseFunc:
    def map(self, pred, act, w, o, model):
        error = act[0] - pred[0]
        error = error ** 2 if error < 0 else 2 * (error ** 2)
        return [error, 1]

    def reduce(self, l, r):
        return [l[0] + r[0], l[1] + r[1]]

    def metric(self, l):
        # Use Java API directly
        import java.lang.Math as math

        return math.sqrt(l[0] / l[1])


# Upload the custom metric
custom_mm_func = h2o.upload_custom_metric(CustomRmseFunc, func_name="rmse_weighted", func_file="mm_rmse.py")

# # Train GBM model with custom metric
# gbm_custom_mm = H2OGradientBoostingEstimator(
#     model_id="custom_delivery_model_mm",
#     ntrees=50,
#     max_depth=5,
#     score_each_iteration=True,
#     stopping_metric="custom",
#     stopping_tolerance=0.1,
#     stopping_rounds=5,
#     distribution="gaussian",
#     custom_metric_func=custom_mm_func,
# )
# gbm_custom_mm.train(y="delivery", x=ind_vars, training_frame=train_h2o, validation_frame=test_h2o)

# # Predict
# predictions_custom_mm = gbm_custom_mm.predict(test_data=test_h2o).as_data_frame()
# deliveries, late, early = evaluate(data_val_treated, predictions_custom_mm)

# # Evalute and print summary
# print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
# print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom_mm.predict)}')

nfolds = 5

gbm_custom_cmm = H2OGradientBoostingEstimator(
    model_id="custom_delivery_model_cmm",
    ntrees=1000,
    score_each_iteration=True,
    stopping_metric="custom",
    stopping_tolerance=0.1,
    stopping_rounds=10,
    distribution="custom",
    custom_metric_func=custom_mm_func,
    custom_distribution_func=distribution_ref,
    max_runtime_secs=30 * 60,
    nfolds=nfolds,
    fold_assignment='Modulo',
    keep_cross_validation_predictions=True
)


# GBM hyperparameters
gbm_params = {
    'learn_rate': [i * 0.01 for i in range(1, 30)],
    'max_depth': list(range(1, 20)),
    'sample_rate': [i * 0.1 for i in range(5, 11)],
    'col_sample_rate': [i * 0.1 for i in range(1, 11)],
}

# Search criteria
search_criteria = {'strategy': 'RandomDiscrete', 'seed': 1, 'max_runtime_secs': 8 * 60 * 60}

# Train and validate a random grid of GBMs
gbm_grid = H2OGridSearch(
    model=gbm_custom_cmm, grid_id='gbm_grid', hyper_params=gbm_params, search_criteria=search_criteria
)

# Train GBM model with custom metric and distribution

gbm_grid.train(y="delivery", x=ind_vars, training_frame=train_h2o, validation_frame=test_h2o)

import ipdb

ipdb.set_trace()

grid_id = gbm_grid.grid_id
old_grid_model_count = len(gbm_grid.model_ids)

# Save the grid
saved_path = h2o.save_grid('./gbm_grid_second', grid_id)

gbm_gridperf = gbm_grid.get_grid(sort_by='rmse', decreasing=False)
best_gbm = gbm_gridperf.models[0]

model_path = h2o.save_model(best_gbm, force=True)
os.rename(model_path, model_path + '_' + time.strftime("%Y%m%d-%H%M%S"))

# Predict
predictions_custom_cmm = best_gbm.predict(test_data=test_h2o).as_data_frame()
deliveries, late, early = evaluate(data_val_treated, predictions_custom_cmm)

# Evalute and print summary
rmse_val = rmse_weighted(data_val_treated.delivery.values, predictions_custom_cmm.predict)
print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
print(f'RMSE Weighted: {rmse_val}')

# Train a stacked ensemble using the GBM grid
ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_gbm_grid", base_models=gbm_grid.model_ids)
ensemble.train(x=ind_vars, y='delivery', training_frame=train_h2o, validation_frame=test_h2o)

# Predict
predictions_custom_cmm = best_gbm.predict(test_data=test_h2o).as_data_frame()
deliveries, late, early = evaluate(data_val_treated, predictions_custom_cmm)

predictions_pred = best_gbm.predict(test_data=pred_h2o).as_data_frame()
pd.DataFrame({'prediction': predictions_pred.predict}).to_csv(
    f'data_to_predict_h2o_{rmse_val}_{late/deliveries}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
)

h2o.cluster().shutdown(prompt=True)
