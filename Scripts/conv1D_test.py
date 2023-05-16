import tensorflow as tf
from keras.models import Model
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input
import numpy as np
import pandas as pd
from sklearn.datasets import lo
from keras.utils import to_categorical

WINDOW_SIZE = 20


X, y = load_iris(return_X_y=True)

X_conv = []
for t in range(WINDOW_SIZE, X.shape[0]):
    X_conv_single = []
    for iFt in range(X.shape[1]):
        X_conv_single.append(X[t-WINDOW_SIZE: t, iFt].tolist())
    X_conv.append(X_conv_single)

X_conv = np.array(X_conv)
X_conv = np.transpose(X_conv, axes=(0, 2, 1))
y = y[WINDOW_SIZE:]
y = to_categorical(y)

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
