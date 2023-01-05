import sklearn.base
import tabpfn as tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
import xgboost
from keras.losses import mean_squared_error
from datetime import datetime as dt

X = np.random.rand(100, 2)
y = np.round(X[:, 0] + X[:, 1] * X[:,0] + 4 * X[:,1]**2).astype(int)

X_test = np.random.rand(10, 2)
y_test = np.round(X_test[:, 0] + X_test[:, 1] * X_test[:,0] + 4 * X_test[:,1]**2).astype(int)

xgb_params = {"n_estimators": 100,
          "max_depth": 5,
          "eta": 0.05,
          "subsample": 0.7,
          "colsample_bytree": 0.8}



class window_classifier(sklearn.base.BaseEstimator):
    def __init__(self, model=None, window_size=30, **kwargs):
        super(window_classifier, self).__init__()
        self.model = model(**kwargs)
        self.window_size = window_size
        self.X = None
        self.y = None
        self.y_pred = None

    def fit(self, X, y, **fit_params):
        self.X = X
        self.y = y
        if "keras" in str(self.model.__class__):
            return self.model.fit(X, y, **fit_params)
        else:
            self.model.fit(X, y, **fit_params)
            return self.model

    def window(self, X, y, dateFrom, dateTo):
        mask = (X.index >= dateFrom) & (X.index <= dateTo)
        X_window = X.loc[mask]
        y_window = y.loc[mask]
        return (X_window, y_window)

    def window_predict(self, X, y):
        y_pred_window = pd.DataFrame()
        for i in range(self.window_size - 1, len(X) - 1):
            # fit on window
            d = X.index[i]
            X_window, y_window = self.window(X, y, d - dt.timedelta(days=window_size - 1), d)
            # xgb_window = xgboost.XGBRegressor(**opt_params)
            self.model.fit(X_window, y_window)
            # predict next day
            y_pred_window = y_pred_window.append(
                pd.DataFrame(self.model.predict(X.iloc[i + 1, :].values.reshape(1, -1)), index=[X.index[i + 1]]))
        self.y_pred = y_pred_window
        return y_pred_window

    def predict(self, X, **predict_params):
        return self.model.predict(X, **predict_params)

    def compile(self, **compile_params):
        ''' only for keras sub models'''
        self.model.compile(**compile_params)
# xgb
model = window_classifier(xgboost.XGBClassifier, 30, **xgb_params)
model.fit(X, y)
y_pred = model.predict(X)
y_pred_test = model.predict(X_test)
acc_train = accuracy_score(y, y_pred)
acc = accuracy_score(y_test, y_pred_test)

# tf mlp
nn = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(2,)),
    tf.keras.layers.Dense(1)]
)

print(nn.summary())
nn.compile(optimizer="Adam",loss=mean_squared_error)
history = nn.fit(X, y, epochs=100)
plt.plot(history.history["loss"])

nn_params = {"layers":[
    tf.keras.layers.Dense(8, input_shape=(2,)),
    tf.keras.layers.Dense(1)]}

model = window_classifier(tf.keras.Sequential, 30, **nn_params)
model.compile(optimizer="Adam",loss=mean_squared_error)
history = model.fit(X, y, epochs=100)
plt.plot(history.history["loss"], marker="*")
plt.show()

# tab pfn
classifier = tp.TabPFNClassifier()
classifier.fit(X, y)
y_pred = classifier.predict(X)

y_pred_test = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred_test)
plt.scatter(y_test, y_pred_test)

plt.scatter(X[:, 0] + X[:, 1] * X[:,0] + 4 * X[:,1]**2, y)