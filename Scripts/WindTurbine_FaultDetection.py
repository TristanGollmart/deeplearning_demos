import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import datetime as dt
from WindTurbine_preprocessing import preprocess
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import norm

PROBABILITY_CUTOFF = 0.1  # mark as faulty if probability to be of the distribution is smaller than this
PROBABILITY_CUTOFF_SEVERE = 0.6  # mark as severe if average probability over one week exceeds this value

def getSevereFaults(isfaulty):
    # detect severe fault from single faults
    # severe fault is a fault lasting for at least one week = 6 * 24 * 7 measurement points
    nIntervalsPerWeek = 6 * 24 * 7  # 10 minute slots
    severeFault = np.zeros(len(isfaulty))

    for i in range(nIntervalsPerWeek, len(isfaulty)):
        severeFault[i] = np.average(isfaulty.iloc[i - nIntervalsPerWeek:i]) > PROBABILITY_CUTOFF_SEVERE
    return pd.Series(data=severeFault, index=isfaulty.index)

# Preprocessing data
df = pd.read_csv(r'..\data\Wind_FaultDetection\wind_data.csv')

X_base, y_base, X_test, y_test = preprocess(df)

# Visualization
y_base.plot()
y_test.plot()
# sn.pairplot(df)
plt.show()

# ------------ modelling ---------------

# LR
# Assumption yt ~ p(yt|xt) independent of y(t-1) given x(t), no further autocorrelation to be included here
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

# GP

GPR = GaussianProcessRegressor(kernel='rbf')
GPR.fit(X_base, y_base)
y_pred_test, sig_pred_test = GPR.predict(X_test, return_std=True)




print("finished")
