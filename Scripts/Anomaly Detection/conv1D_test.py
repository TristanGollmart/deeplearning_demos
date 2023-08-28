from tensorflow import expand_dims
from keras.models import Model
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, Conv1DTranspose
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 20


X, y = load_iris(return_X_y=True)

sc = StandardScaler()
X = sc.fit_transform(X)

x_rec = X[:, 0].reshape(-1, 1)

X_conv = []
x_rec_conv = []
for t in range(WINDOW_SIZE, X.shape[0]):
    X_conv_single = []
    x_rec_sequence = []
    for iFt in range(X.shape[1]):
        X_conv_single.append(X[t-WINDOW_SIZE: t, iFt].tolist())

    x_rec_sequence.append(x_rec[t-WINDOW_SIZE:t, 0].tolist())
    X_conv.append(X_conv_single)
    x_rec_conv.append(x_rec_sequence)

X_conv = np.array(X_conv)
X_conv = np.transpose(X_conv, axes=(0, 2, 1))
x_rec_conv = np.array(x_rec_conv)
x_rec_conv = np.transpose(x_rec_conv, axes=(0,2,1))
y = y[WINDOW_SIZE:]
y = to_categorical(y)


# reconstructor
input = Input(shape=(WINDOW_SIZE, x_rec.shape[1]))
conv1d = Conv1D(8, kernel_size=7, padding="valid", activation='relu')(input)
conv1d = Dropout(0.2)(conv1d)
conv1d = Conv1D(16, kernel_size=5, padding="valid", activation='relu')(conv1d)
conv1d = Dropout(0.2)(conv1d)
conv1d = Conv1D(32, kernel_size=3, padding="valid", activation='relu')(conv1d)
conv1d = Dropout(0.2)(conv1d)

#     reconstruction
conv1d = Conv1DTranspose(32, kernel_size=3, padding='valid', activation='relu')(conv1d)
conv1d = Dropout(0.2)(conv1d)
conv1d = Conv1DTranspose(16, kernel_size=5, padding='valid', activation='relu')(conv1d)
conv1d = Dropout(0.2)(conv1d)
conv1d = Conv1DTranspose(8, kernel_size=7, padding='valid', activation='relu')(conv1d)

flat = Flatten()(conv1d)
output = Dense(WINDOW_SIZE)(flat)
output = expand_dims(output, 2)

model = Model(input, output)
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mse"])
history = model.fit(x_rec_conv, x_rec_conv, epochs=100)

x_rec_pred = model.predict(x_rec_conv)
plt.plot(x_rec_conv[0, :, 0])
plt.plot(x_rec_pred[0, :, 0])
plt.show()

plt.plot(x_rec_conv[:, 0, 0])
plt.plot(x_rec_pred[:, 0, 0])
plt.show()
# predictor

print(X_conv.shape)

nClasses = 3

input = Input(shape=(WINDOW_SIZE, X.shape[1]))
conv1d = Conv1D(1, 7, padding="causal", activation='relu')(input)
flat = Flatten()(conv1d)
output = Dense(nClasses, activation='softmax')(flat)

model = Model(input, output)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"])
history = model.fit(X_conv, y, epochs=100)

print(history.history["loss"])
