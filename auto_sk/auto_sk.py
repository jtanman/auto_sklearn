import autosklearn.regression
import feather
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import os
import time
import pickle


def mse_weighted(y, yhat):
    # weighted error squaring the error and then doubling it if the actual delivery is late/greater
    err = np.mean(np.where(y > yhat, 2 * ((y - yhat) ** 2), (y - yhat) ** 2))
    return err


def rmse_weighted(y, yhat):
    # weighted error squaring the error and then doubling it if the actual delivery is late/greater
    # sum = 0
    # for i in range(len(y)):
    #     y_temp = y[i]
    #     yhat_temp = yhat[i]
    #     if y_temp > yhat_temp:
    #         sum += 2 * ((y_temp - yhat_temp) ** 2)
    #     else:
    #         sum += (y_temp - yhat_temp) ** 2
    
    # return np.sqrt(sum / len(y))
    sample_weight = np.where(y > yhat, 2, 1)
    output_errors = np.sqrt(np.average((y_true - y_pred) ** 2, axis=0,
                               weights=sample_weight))
    return err


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
        name="mse_weighted",
        score_func=rmse_weighted,
        optimum=0,
        worst_possible_result=MAXINT,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )

    mean_squared_error = make_scorer('mean_squared_error',
                                 sklearn.metrics.mean_squared_error,
                                 optimum=0,
                                 worst_possible_result=MAXINT,
                                 greater_is_better=False)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=10*60,
        per_run_time_limit=5*60,
        ml_memory_limit=9216,
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

    print(automl.show_models())
    print(automl.sprint_statistics())

    import ipdb; ipdb.set_trace()

    # with open('automl_20200811-013151.pickle', 'rb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     automl = pickle.load(f)

    print(f'RMSE train: {np.sqrt(sklearn.metrics.mean_squared_error(y_train, automl.predict(X_train)))}')

    yhat = automl.predict(X_test)

    print(f'RMSE: {np.sqrt(sklearn.metrics.mean_squared_error(y_test, yhat))}')
    print(f'RMSE Weighted: {rmse_weighted(y_test, yhat)}')
    print(f'MAE: {sklearn.metrics.mean_absolute_error(y_test, yhat)}')
    print(f'r2: {	sklearn.metrics.r2_score(y_test, yhat)}')

    import ipdb; ipdb.set_trace()

    data_pred_treated = feather.read_dataframe('./data_pred_treated.feather')
    pd.DataFrame({'prediction': automl.predict(data_pred_treated)}).to_csv('data_to_predict.csv')

    import ipdb; ipdb.set_trace()

    return automl
