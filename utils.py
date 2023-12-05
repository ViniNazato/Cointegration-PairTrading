import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, norm
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss


def checkresiduals(fitted_model, plot: bool = True, lb_lag=20):

    d = np.maximum(fitted_model.loglikelihood_burn, fitted_model.nobs_diffuse)

    if hasattr(fitted_model.data, 'dates') and fitted_model.data.dates is not None:
        ix = fitted_model.data.dates[d:]
    else:
        ix = np.arange(fitted_model.nobs - d)

    residuals = pd.Series(
        fitted_model.filter_results.standardized_forecasts_error[0, d:],
        index=ix)

    ljungbox = acorr_ljungbox(residuals, lags=lb_lag)
    jarquebera = jarque_bera(resids=residuals)

    print()
    if plot:
        _, ax = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)']],
                                   layout='constrained', figsize=(10, 5))

        # Residuals vs. Time plot
        residuals.plot(ax=ax['a)'], color='k', linewidth=1.0)
        ax['a)'].set_title('ARIMA Residuals')
        ax['a)'].hlines(0, ix[0], ix[-1], alpha=0.4, color='k',)

        # Residuals ACF
        plot_acf(residuals, lags=40, ax=ax['b)'], auto_ylims=True, marker='.')
        ax['b)'].set_title('ACF of Residuals')

        # Histogram of residuals
        ax['c)'].hist(residuals, bins=20, density=True, alpha=0.7, color='g')
        ax['c)'].set_title('Histogram of Residuals')

        kde = gaussian_kde(residuals)
        xlim = (-1.96*2, 1.96*2)
        x = np.linspace(xlim[0], xlim[1])
        ax['c)'].plot(x, kde(x), label='KDE')
        ax['c)'].plot(x, norm.pdf(x), label='N(0,1)',
                      color='darkorange', linestyle='-')
        ax['c)'].set_xlim(xlim)
        ax['c)'].legend()
        ax['c)'].set_title('Histogram plus estimated density')

        plt.tight_layout()

    print('=========== Ljung-Box ===========')
    print('Ljung-Box Test Statistic:', ljungbox.lb_stat[lb_lag])
    print('Ljung-Box p-value:', ljungbox.lb_pvalue[lb_lag])
    print('=========== Jarque-Bera ===========')
    print('Jarque Bera Test Statistic:', jarquebera[0])
    print('Jarque Bera p-value:', jarquebera[1])


def one_step_arima(arima_fitted, outsample):

    forecasts = []
    model = arima_fitted

    for t in range(outsample.shape[0]):
        yhat = model.forecast(steps=1)
        forecasts.append(yhat[0])
        model = model.append(outsample.iloc[t:t+1], refit=False)

    return forecasts


def metrics(outsample, predictions):

    mae = mean_absolute_error(outsample, predictions)
    mse = mean_squared_error(outsample, predictions)
    mape = mean_absolute_percentage_error(outsample, predictions)
    rmse = np.sqrt(mse)

    return pd.DataFrame([
        {"Metric": "RMSE", "Value": rmse},
        {"Metric": "MAE", "Value": mae},
        {"Metric": "MAPE", "Value": mape},
    ])


def adf_test(x, signif='5%', verbose=False):
    """
    Perform the Augmented Dickey-Fuller (ADF) test for stationarity on a given time series.

    Parameters:
    - x (pd.Series or array-like): The time series data to be tested for stationarity.
    - signif (str, optional): The significance level for the critical values, default is '5%'.
    - verbose (bool, optional): If True, print detailed information about the test results for each regression type.

    Returns:
    pd.DataFrame: A DataFrame summarizing the results of the ADF test for different regression types. 
                  Columns include Test_Type, Test_Statistic, Critical_Value, p-value, Used_lags, and Conclusion.

    Notes:
    The ADF test is used to assess the stationarity (Unit  Root) of a time series. The test is performed for three regression types:
    1. 'n': None (No trend or constant)
    2. 'c': Constant
    3. 'ct': Constant/Trend

    The conclusion is based on whether the test statistic is less than the critical value at the specified significance level.
    If the test statistic is less than the critical value, the time series is considered stationary; otherwise, it is considered non-stationary.
    """

    regression_types = ['n', 'c', 'ct']  # List of regression types
    # Labels for regression types
    labels = ['None', 'Constant', 'Constant/Trend']
    test_results = pd.DataFrame(
        columns=['Test_Type', 'Test_Statistic', 'Critical_Value', 'p-value', 'Used_lags', 'Conclusion'])

    for regression_type, label in zip(regression_types, labels):
        adf = adfuller(x, regression=regression_type,
                       autolag='BIC', maxlag=20)
        if verbose:
            print(f'-------------{label}--------------')
            print(f'ADF Statistic:{adf[0]}')
            print(f'p-value: {adf[1]}')
            print(f'Used Lags: {adf[2]}')
            print(f'Critical Values: {adf[4]}')
            if adf[0] < adf[4].get('5%'):
                print('--> Stationary')
            else:
                print('--> Non-Stationary')

        if adf[0] < adf[4].get('5%'):
            conclusion = 'Stationary'
        else:
            conclusion = 'Non-Stationary'

        append = pd.DataFrame({'Test_Type': label,
                               'Test_Statistic': adf[0],
                               'p-value': adf[1],
                               'Used_lags': adf[2],
                               'Critical_Value': adf[4][signif],
                               'Conclusion': conclusion},
                              index=[0])

        test_results = pd.concat(
            (test_results, append), ignore_index=True)

    return test_results


def kpss_test(x, signif='5%', verbose=False):
    """
    Perform the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity on a given time series.

    Parameters:
    - x (pd.Series or array-like): The time series data to be tested for stationarity.
    - signif (str, optional): The significance level for the critical values, default is '5%'.
    - verbose (bool, optional): If True, print detailed information about the test results for each regression type.

    Returns:
    pd.DataFrame: A DataFrame summarizing the results of the KPSS test for different regression types. 
                  Columns include Test_Type, Test_Statistic, Critical_Value, p-value, Used_lags, and Conclusion.

    Notes:
    The KPSS test is used to assess the stationarity of a time series. The test is performed for two regression types:
    1. 'c': Constant
    2. 'ct': Constant/Trend

    The conclusion is based on whether the test statistic is less than the critical value at the specified significance level.
    If the test statistic is less than the critical value, the time series is considered stationary; otherwise, it is considered non-stationary.
    """

    regression_types = ['c', 'ct']  # List of regression types
    labels = ['Constant', 'Constant/Trend']  # Labels for regression types

    test_results = pd.DataFrame(
        columns=['Test_Type', 'Test_Statistic', 'Critical_Value', 'p-value', 'Used_lags', 'Conclusion'])

    for regression_type, label in zip(regression_types, labels):
        kpss_ = kpss(x, regression=regression_type)
        if verbose:
            print(F'-------------{label}--------------')
            print(f'KPSS Statistic:{kpss_[0]}')
            print(f'p-value: {kpss_[1]}')
            print(f'Used Lags: {kpss_[2]}')
            print(f'Critical Values: {kpss_[3]}')

            if kpss_[0] < kpss_[3].get('5%'):
                print('--> Stationary')
            else:
                print('--> Non-Stationary')

        if kpss_[0] < kpss_[3].get('5%'):
            conclusion = 'Stationary'
        else:
            conclusion = 'Non-Stationary'

        append = pd.DataFrame({'Test_Type': label,
                               'Test_Statistic': kpss_[0],
                               'p-value': kpss_[1],
                               'Used_lags': kpss_[2],
                               'Critical_Value': kpss_[3][signif],
                               'Conclusion': conclusion}, index=[0])

        test_results = pd.concat(
            (test_results, append), ignore_index=True)

    return test_results
