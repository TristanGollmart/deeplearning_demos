import keras
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
    x_rec_conv.append(x_rec_sequence)

x_rec_conv = np.array(x_rec_conv)
x_rec_conv = np.transpose(x_rec_conv, axes=(0, 2, 1))
y = y[WINDOW_SIZE:]
y = to_categorical(y)


class TSReconstructor(keras.Model):
    def __init__(self, nFeatures=1):
        super(TSReconstructor, self).__init__()
        self.Dropout = Dropout(0.2)
        self.Flatten = Flatten()
        self.conv1D_1 = Conv1D(8, kernel_size=7, padding="valid", input_shape=(WINDOW_SIZE, nFeatures), activation='relu')
        self.conv1D_2 = Conv1D(16, kernel_size=7, padding="valid", activation='relu')
        self.conv1D_3 = Conv1D(32, kernel_size=7, padding="valid", activation='relu')

        self.convT1D_1 = Conv1DTranspose(32, kernel_size=3, padding='valid', activation='relu')
        self.convT1D_2 = Conv1DTranspose(16, kernel_size=3, padding='valid', activation='relu')
        self.convT1D_3 = Conv1DTranspose(8, kernel_size=3, padding='valid', activation='relu')

        self.Classifier = Dense(WINDOW_SIZE)

    def call(self, input):
        conv = self.conv1D_1(input)
        conv = self.Dropout(conv)
        conv = self.conv1D_2(conv)
        conv = self.Dropout(conv)
        conv = self.conv1D_3(conv)
        conv = self.Dropout(conv)
        deconv = self.convT1D_1(conv)
        deconv = self.Dropout(deconv)
        deconv = self.convT1D_2(deconv)
        deconv = self.Dropout(deconv)
        deconv = self.convT1D_3(deconv)
        flat = self.Flatten(deconv)
        output = self.Classifier(flat)
        return expand_dims(output, 2)



# reconstructor
TR = TSReconstructor()
input = Input(shape=(WINDOW_SIZE, x_rec.shape[1]))
output = TR(input)
model = Model(input, output)

# conv1d = Conv1D(8, kernel_size=7, padding="valid", activation='relu')(input)
# conv1d = Dropout(0.2)(conv1d)
# conv1d = Conv1D(16, kernel_size=5, padding="valid", activation='relu')(conv1d)
# conv1d = Dropout(0.2)(conv1d)
# conv1d = Conv1D(32, kernel_size=3, padding="valid", activation='relu')(conv1d)
# conv1d = Dropout(0.2)(conv1d)
#
# #     reconstruction
# conv1d = Conv1DTranspose(32, kernel_size=3, padding='valid', activation='relu')(conv1d)
# conv1d = Dropout(0.2)(conv1d)
# conv1d = Conv1DTranspose(16, kernel_size=5, padding='valid', activation='relu')(conv1d)
# conv1d = Dropout(0.2)(conv1d)
# conv1d = Conv1DTranspose(8, kernel_size=7, padding='valid', activation='relu')(conv1d)
#
# flat = Flatten()(conv1d)
# output = Dense(WINDOW_SIZE)(flat)
# output = expand_dims(output, 2)
#model = Model(input, output)

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mse"])
history = model.fit(x_rec_conv, x_rec_conv, epochs=100)

x_rec_pred = model.predict(x_rec_conv)
plt.plot(x_rec_conv[0, :, 0])
plt.plot(x_rec_pred[0, :, 0])
plt.show()

plt.plot(x_rec_conv[:, 0, 0], label='First value of sequence')
plt.plot(x_rec_pred[:, 0, 0], label='First value reconstructed')
plt.legend()
plt.show()

model.save(r'..\models\TimeseriesReconstructor')



