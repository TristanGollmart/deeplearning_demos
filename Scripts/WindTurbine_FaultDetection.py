import datetime

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import datetime as dt
from WindTurbine_preprocessing import preprocess
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import norm

PROBABILITY_CUTOFF = 0.1  # mark as faulty if probability to be of the distribution is smaller than this
PROBABILITY_CUTOFF_SEVERE = 0.6  # mark as severe if average probability over one week exceeds this value
TRAIN_MODEL = True
sModel = 'LR_lag'  # 'TR' #'GP'

def getSevereFaults(isfaulty):
    # detect severe fault from single faults
    # severe fault is a fault lasting for at least one week = 6 * 24 * 7 measurement points

    nHourInterval = (isfaulty.index[1] - isfaulty.index[0]).total_seconds() / 3600

    nIntervalsPerWeek = int(24 * 7 / nHourInterval)  # 10 minute slots
    severeFault = np.zeros(len(isfaulty))

    for i in range(nIntervalsPerWeek, len(isfaulty)):
        severeFault[i] = np.average(isfaulty.iloc[i - nIntervalsPerWeek:i]) > PROBABILITY_CUTOFF_SEVERE
    return pd.Series(data=severeFault, index=isfaulty.index)

def add_lag(X, y, lag=1):
    X_lag = X.shift(lag)
    X_lag.columns = [col + '_lag' for col in X.columns]
    y_lag = pd.DataFrame(y.shift(lag))
    X = pd.concat([X, X_lag, y_lag, y], axis=1).dropna()
    return X.iloc[:, :-1], X.iloc[:, -1]
# Preprocessing data
df = pd.read_csv(r'..\data\Wind_FaultDetection\wind_data.csv')

X_base, y_base, X_test, y_test = preprocess(df)

# Visualization
#y_base.plot(label='scaled target data first year')
#y_test.plot(label='scaled target data second year')
#plt.legend()
#plt.show()

# ------------ modelling ---------------

if sModel == 'LR':
    # LR
    # Assumption yt ~ p(yt|xt) independent of y(t-1) given x(t), no further autocorrelation to be included here
    # Use LR instead of ARIMA since no partial auto correlation, and Arima would fit also outliers too well
    LR = LinearRegression()
    LR.fit(X_base, y_base)
    y_pred_base = LR.predict(X_base)
    mse_base = mean_squared_error(y_base, y_pred_base)
    print('mse score on base set: {:.3f}'.format(mse_base))

    y_pred_test = LR.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print('mse score on test set: {:.3f}'.format(mse_test))

    residuals = y_test - y_pred_test

    # ---------------------------
    # Add here normality check of residuals
    # histogram and anderson test
    # ---------------------------

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(residuals, alpha=0.2, color='g', label='residuals')

    # assumption residuals epsilon are white noise -> faults at +-3 sigma of epsilon are assumed out of distribution
    std = np.std(residuals)

    isfaulty = pd.Series(data=(norm.pdf(residuals, loc=0, scale=std) < PROBABILITY_CUTOFF), index=y_test.index)
    ax2.plot(isfaulty, alpha=0.6, color='orange', label='faulty')
    severeFault = getSevereFaults(isfaulty)

    ax2.plot(severeFault, alpha=0.7, color='red', label='severe')
    fig.legend(loc='upper right')
    plt.show()

if sModel == 'LR_lag':
    # ...with lag
    X_base_lag, y_base = add_lag(X_base, y_base, lag=24*7*6)
    X_test_lag, y_test = add_lag(X_test, y_test, lag=24*7*6)
    LR = LinearRegression()
    LR.fit(X_base_lag, y_base)
    y_pred_base = LR.predict(X_base_lag)
    mse_base = mean_squared_error(y_base, y_pred_base)
    print('mse score on base set: {:.3f}'.format(mse_base))

    y_pred_test = LR.predict(X_test_lag)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print('mse score on test set: {:.3f}'.format(mse_test))

    residuals = y_test - y_pred_test

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(residuals, alpha=0.2, color='g', label='residuals')

    # assumption residuals epsilon are white noise -> faults at +-3 sigma of epsilon are assumed out of distribution
    std = np.std(residuals)

    isfaulty = pd.Series(data=(norm.pdf(residuals, loc=0, scale=std) < PROBABILITY_CUTOFF), index=y_test.index)
    ax2.plot(isfaulty, alpha=0.4, color='orange', label='faulty')
    severeFault = getSevereFaults(isfaulty)

    ax2.plot(severeFault, alpha=0.7, color='red', label='severe')
    fig.legend(loc='upper right')
    plt.show()

if sModel == 'GP':
    # GP
    X_base_hourly = X_base.loc[X_base.index.minute == 0]
    y_base_hourly = y_base.loc[y_base.index.minute == 0]
    X_test_hourly = X_test.loc[X_test.index.minute == 0]
    y_test_hourly = y_test.loc[y_test.index.minute == 0]

    #X_base_hourly.plot()
    #X_test_hourly.plot()
    GPR = GaussianProcessRegressor(kernel=kernels.RBF(length_scale=1., length_scale_bounds='fixed'))
    GPR.fit(X_base_hourly, y_base_hourly)
    y_pred, sig_pred = GPR.predict(X_base_hourly, return_std=True)
    y_pred, sig_pred = pd.Series(y_pred, index=X_base_hourly.index), pd.Series(sig_pred,
                                                                                              index=X_base_hourly.index)
    y_pred_test, sig_pred_test = GPR.predict(X_test_hourly, return_std=True)
    y_pred_test, sig_pred_test = pd.Series(y_pred_test, index=X_test_hourly.index), pd.Series(sig_pred_test, index=X_test_hourly.index)
    prob = norm.pdf(y_test_hourly, loc=y_pred_test, scale=sig_pred_test)
    mse_base = mean_squared_error(y_base_hourly, y_pred)
    print('mse score on base set: {:.3f}'.format(mse_base))
    mse_test = mean_squared_error(y_test_hourly, y_pred_test)
    print('mse score on test set: {:.3f}'.format(mse_test))

    isfaulty = pd.Series(data=(prob < PROBABILITY_CUTOFF), index=y_test_hourly.index)

    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.plot(y_pred, alpha=0.2, color='g', label='mu predicted')
    ax1.plot(y_base_hourly, alpha=0.2, color='yellow', label='data')
    ax1.plot(y_pred_test, alpha=0.2, color='g', label='mu predicted')
    ax1.plot(y_test_hourly, alpha=0.2, color='yellow', label='data')

    severeFault = getSevereFaults(isfaulty)
    ax2.plot(severeFault, alpha=0.7, color='red', label='severe')
    fig.legend(loc='upper right')
    plt.show()

if sModel == 'DBSCAN':
    # DBSCAN
    from sklearn.cluster import DBSCAN
    from collections import Counter

    X_base_hourly = X_base.loc[X_base.index.minute == 0]
    y_base_hourly = y_base.loc[y_base.index.minute == 0]
    X_test_hourly = X_test.loc[X_test.index.minute == 0]
    y_test_hourly = y_test.loc[y_test.index.minute == 0]
    X_hourly = pd.concat([X_base_hourly, X_test_hourly], axis=0)
    y_hourly = pd.concat([y_base_hourly, y_test_hourly], axis=0)
    df = pd.concat([X_hourly, y_hourly], axis=1)

    # ToDo Find eps via Rank-list of kNN
    DB = DBSCAN(eps=0.6, min_samples=100)
    clusters = DB.fit(df)
    counts = Counter(clusters.labels_)

    # outliers can be ascribed to faulty target since covariates are assumed correct
    outlier_labels = [-1]
    outlier_index = [i for i, label in enumerate(clusters.labels_) if label in outlier_labels]
    outliers = df.iloc[outlier_index, :]



    # Count outliers per week
    nItemsPerWeek = 24 * 7
    threashold_outliers = 40
    severe_faults = np.zeros(len(df))
    for i in range(len(df) - nItemsPerWeek):
        dstart = df.index[i]
        dend = dstart + datetime.timedelta(days=7)
        outliers_week = outliers.loc[(outliers.index > dstart) & (outliers.index < dend)]
        if len(outliers_week) > threashold_outliers:
            severe_faults[i] = 1

    plt.plot(df.iloc[:, -1], alpha=0.5, label='data')
    plt.scatter(outliers.index, outliers.iloc[:, -1], color='red', marker='*', label='outliers')
    pd.Series(severe_faults, index= df.index).plot(color='red', label='severe faults')
    plt.title('DBSCAN outliers')
    plt.legend()
    plt.show()


if sModel == 'TR':
    # Reconstructor: Either output -> output or input -> output
    # Advantage: not reliant on input measurements X
    from Reconstructor import TSReconstructor, transform_input_1d, transform_input_2d
    from tensorflow import expand_dims
    from keras.models import Model
    from keras.layers import Input
    from keras.models import load_model

    WINDOW_SIZE = 24 * 7 # 1 week

    X_base_hourly = X_base.loc[X_base.index.minute == 0]
    y_base_hourly = y_base.loc[y_base.index.minute == 0]
    X_test_hourly = X_test.loc[X_test.index.minute == 0]
    y_test_hourly = y_test.loc[y_test.index.minute == 0]

    y_rec = transform_input_1d(y_base_hourly, WINDOW_SIZE)
    y_rec_test = transform_input_1d(y_test_hourly, WINDOW_SIZE)
    x_rec = transform_input_2d(X_base_hourly.values, WINDOW_SIZE)
    x_rec_test = transform_input_2d(X_test_hourly.values, WINDOW_SIZE)

    if TRAIN_MODEL:
        TR = TSReconstructor(seq_length=WINDOW_SIZE, nFeatures=x_rec.shape[-1], padding_conv="causal")
        input = Input(shape=(WINDOW_SIZE, x_rec.shape[-1]))
        output = TR(input)
        model = Model(input, output)

        ypred_ = model.predict(x_rec)

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mse"])
        history = model.fit(x_rec, y_rec, epochs=100)
        model.save(r'..\models\TimeseriesReconstructor')
    else:
        model = load_model(r'..\models\TimeseriesReconstructor')

    model.summary()
    y_rec_pred_train = model.predict(x_rec)
    mse_base = mean_squared_error(y_rec[:, :, 0], y_rec_pred_train[:, :, 0])
    print('mse score on base set: {:.3f}'.format(mse_base))

    y_rec_pred = model.predict(x_rec_test)
    mse_test = mean_squared_error(y_rec_test[:, :, 0], y_rec_pred[:, :, 0])
    print('mse score on test set: {:.3f}'.format(mse_test))

    residuals = y_rec_test[:, -1, 0] - y_rec_pred[:, -1, 0]

    plt.plot(y_rec_test[0, :, 0], label='First sequence')
    plt.plot(y_rec_pred[0, :, 0], label='First sequence reconstructed')
    plt.title('Single Sequence reconstructed')
    plt.legend()
    plt.show()

    plt.plot(y_rec_test[:, 0, 0], color='green', label='First value of sequence')
    plt.plot(y_rec_pred[:, 0, 0], color = 'yellow', label='First value reconstructed')
    plt.plot(residuals, alpha = 0.2, color='red', label='residuals')
    plt.title('Time Serie Reconstructed')
    plt.legend()
    plt.show()

    plt.plot(y_rec_test[:, -1, 0], label='Last value of sequence')
    plt.plot(y_rec_pred[:, -1, 0], label='Last value reconstructed')
    plt.title('Single Sequence reconstructed')
    plt.legend()
    plt.show()

    plt.plot(residuals, label='residuals')
    plt.title('TR Resdiuals')
    plt.legend()
    plt.show()

print("finished")
