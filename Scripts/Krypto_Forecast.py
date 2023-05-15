import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Input
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

WINDOW_SIZE = 20

df = pd.read_csv(r'..\data\Crypto_Prediction\Kraken_BTCUSD_d.csv', header = 1)


df['returns'] = df['Open'] / df['Open'].shift(-1) -1
df['High returns'] = df['High'] / df['High'].shift(-1) -1
df = df.dropna()
print(df)

X = []
Y = []

for i in range(len(df['returns']) - WINDOW_SIZE):
    Y.append(df['returns'].iloc[i])
    X.append(df[['returns', 'High returns']].iloc[i+1: i+WINDOW_SIZE+1, :][::-1].values.tolist())

X = np.array(X)
X = X.reshape((-1, WINDOW_SIZE, X.shape[2]))
Y = np.array(Y)
input = Input((WINDOW_SIZE, X.shape[2]))
lstm = LSTM(1, input_shape=(WINDOW_SIZE, X.shape[2]))(input)
model = Model(input, lstm)


model.compile(optimizer='adam', loss='mse')
history = model.fit(X, Y, batch_size=32, epochs = 10)
ypred = model.predict(X)
ypred = ypred.reshape(-1)
ypred = np.append(ypred, np.zeros(WINDOW_SIZE))
df['return_Predictions'] = ypred
df['Predictions'] = df['Open'].shift(-1) * (df['return_Predictions'] + 1)

df['Naive Model'] = df['Open'].shift(-1)

df = df.dropna()

mse_naive = ((df['Naive Model'] - df['Open']) ** 2).sum() / df['Open'].size
mse_lstm = ((df['Predictions'] - df['Open']) ** 2).sum() / df['Open'].size

years = mdates.YearLocator()
months = mdates.MonthLocator()
yearsfmt = mdates.DateFormatter('%Y')
dates = np.array(df['Date']).astype(np.datetime64)

ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsfmt)
ax.xaxis.set_minor_locator(months)
plt.plot(dates, df['Predictions'])
plt.plot(dates, df['Open'])
plt.show()

print('finished')