import datetime

import feather
import h2o
import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.utils.distributions import CustomDistributionGaussian

from auto_sk.auto_sk import rmse_weighted


h2o.init()

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

data_train_treated = data_train_treated.sample(n=10000, axis=0)

ind_vars = list(data_train_treated.columns)
ind_vars.remove('delivery')

train_h2o = h2o.H2OFrame(data_train_treated)
test_h2o = h2o.H2OFrame(data_val_treated)

gbm_gaussian = H2OGradientBoostingEstimator(
    model_id="delivery_model", ntrees=50, max_depth=5, score_each_iteration=True, distribution="gaussian"
)

gbm_gaussian.train(y="delivery", x=ind_vars, training_frame=train_h2o)

# Predict
predictions = gbm_gaussian.predict(test_data=test_h2o).as_data_frame()
deliveries, late, early = evaluate(data_val_treated, predictions)

print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions.predict)}')

# Define asymmetric loss distribution from Gaussian distribution
class AsymmetricLossDistribution(CustomDistributionGaussian):
    def gradient(self, y, f):
        print(y, f)
        error = (y - f)
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
deliveries, late, early = evaluate(data_val_treated, predictions_custom)

# Evalute and print summary
print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom.predict)}')

print("original vs. custom")
print("prediction mean:", predictions.predict.mean(), predictions_custom.predict.mean())
print("prediction variance:", predictions.predict.var(), predictions_custom.predict.var())
print("residual mean:", predictions.sresidual.mean(), predictions_custom.sresidual.mean())
print("residual variance:", predictions.sresidual.var(), predictions_custom.sresidual.var())

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
        error = error if error < 0 else 2 * error
        return [error * error, 1]

    def reduce(self, l, r):
        return [l[0] + r[0], l[1] + r[1]]

    def metric(self, l):
        # Use Java API directly
        import java.lang.Math as math

        return math.sqrt(l[0] / l[1])


# Upload the custom metric
custom_mm_func = h2o.upload_custom_metric(CustomRmseFunc, func_name="rmse", func_file="mm_rmse.py")

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
    custom_metric_func=custom_mm_func,
)

import ipdb

ipdb.set_trace()
gbm_custom_mm.train(y="delivery", x=ind_vars, training_frame=train_h2o, validation_frame=test_h2o)

# Predict
predictions_custom_mm = gbm_custom_mm.predict(test_data=test_h2o).as_data_frame()
deliveries, late, early = evaluate(data_val_treated, predictions_custom_mm)

# Evalute and print summary
print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom_mm.predict)}')

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
    custom_metric_func=custom_mm_func,
    custom_distribution_func=distribution_ref,
)

# Predict
predictions_custom_cmm = gbm_custom_cmm.predict(test_data=test_h2o).as_data_frame()
deliveries, late, early = evaluate(data_val_treated, predictions_custom_cmm)

# Evalute and print summary
print(f'Deliveries: {deliveries}, % Late: {late/deliveries}')
print(f'RMSE Weighted: {rmse_weighted(data_val_treated.delivery.values, predictions_custom_cmm.predict)}')
