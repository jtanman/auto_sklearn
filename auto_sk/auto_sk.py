import os
import pickle
import time

import autosklearn.regression
import feather
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.validation import _num_samples, check_array, check_consistent_length

def mse_weighted(y, yhat):
    # weighted error squaring the error and then doubling it if the actual delivery is late/greater
    err = np.mean(np.where(y > yhat, 2 * ((y - yhat) ** 2), (y - yhat) ** 2))
    return err


def rmse_weighted(y, y_hat):
    # weighted error squaring the error and then doubling it if the actual delivery is late/greater
    err = np.sqrt(np.mean(np.where(y > y_hat, 2 * ((y - y_hat) ** 2), (y - y_hat) ** 2)))
    return err


def mean_squared_error_weighted(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=False):
    """Mean squared error weighted regression loss
    Read more in the :ref:`User Guide <mean_squared_error>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average'] \
                or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    squared : boolean value, optional (default = True)
        If True returns MSE value, if False returns RMSE value.
    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.612...
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)
    0.708...
    >>> mean_squared_error(y_true, y_pred, squared=False)
    0.822...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825...
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    sample_weight = np.where(y_true > y_pred, 2, 1)
    output_errors = np.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)

    if not squared:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def run():

    # load feather data pre-processed in R and saved

    data_train_treated = feather.read_dataframe('./data_train_treated.feather')
    data_val_treated = feather.read_dataframe('./data_val_treated.feather')

    # data_train_treated = data_train_treated.sample(n=10000, axis=0)

    X_train = data_train_treated.drop('delivery', axis=1)
    y_train = data_train_treated['delivery']

    X_test = data_val_treated.drop('delivery', axis=1)
    y_test = data_val_treated['delivery']

    print("#" * 80)
    print("Use self defined accuracy metric")
    rmse_weighted_scorer = autosklearn.metrics.make_scorer(
        name="rmse_weighted",
        score_func=mean_squared_error_weighted,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=10 * 60,
        per_run_time_limit=5 * 60,
        ml_memory_limit=8192,
        include_preprocessors=['no_preprocessing',],
        tmp_folder=f'{os.getcwd()}/autosklearn_regression_example_tmp',
        output_folder=f'{os.getcwd()}/autosklearn_regression_example_out',
        metric=rmse_weighted_scorer,
        # metric=autosklearn.metrics.mean_squared_error,
    )
    import ipdb; ipdb.set_trace()
    automl.fit(X_train, y_train, X_test=X_test, y_test=y_test, dataset_name='doordash')

    with open(f'automl_{time.strftime("%Y%m%d-%H%M%S")}.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    # with open('automl_20200812-183241.pickle', 'rb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     automl = pickle.load(f)

    import ipdb; ipdb.set_trace()

    print(automl.show_models())
    print(automl.sprint_statistics())

    # print(f'RMSE train: {np.sqrt(sklearn.metrics.mean_squared_error(y_train, automl.predict(X_train)))}')

    yhat = automl.predict(X_test)

    print(f'RMSE: {np.sqrt(sklearn.metrics.mean_squared_error(y_test, yhat))}')
    print(f'RMSE Weighted: {mean_squared_error_weighted(y_test, yhat)}')
    print(f'RMSE Weighted Old: {rmse_weighted(y_test, yhat)}')
    print(f'MAE: {sklearn.metrics.mean_absolute_error(y_test, yhat)}')
    print(f'r2: {	sklearn.metrics.r2_score(y_test, yhat)}')

    data_pred_treated = feather.read_dataframe('./data_pred_treated.feather')
    pd.DataFrame({'prediction': automl.predict(data_pred_treated)}).to_csv(f'data_to_predict_{time.strftime("%Y%m%d-%H%M%S")}.csv')

    import ipdb; ipdb.set_trace()

    return automl
