import keras
import tensorflow as tf
import gzip
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense, MaxPool2D, Conv2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# image transformations


def open_images(filename):
        with gzip.open(filename, 'rb') as file:
            data = file.read()
            return np.frombuffer(data, dtype=np.uint8, offset=16)\
                .reshape(-1, 28, 28)\
                .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, 'rb') as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)

# get fashionMNIST data


X_train = open_images("../data/fashion/train-images-idx3-ubyte.gz")
y_train = open_labels("../data/fashion/train-labels-idx1-ubyte.gz")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")
y_test = open_labels("../data/fashion/t10k-labels-idx1-ubyte.gz")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model using keras functional API
input = tf.keras.Input(shape=(28, 28, 1))
conv1 = Conv2D(10, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))(input)
flattened = Flatten()(conv1)
output = Dense(10, activation='sigmoid')(flattened)

model = Model(input, output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

nData_train = X_train.shape[0]
nData_val = X_val.shape[0]

history = model.fit(
    X_train.reshape(60000, 28, 28, 1),
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=1000
)

print(history)


# Get intermediate outputs after first layer for visualization
from matplotlib import pyplot as plt

model_firstLayer = Model(input, conv1)
output1 = model_firstLayer.predict(X_train[0, :, :].reshape(1,28,28,1))

plt.imshow(X_train[0, :, :])

kernel_number = 7
plt.imshow(output1[0, :, :, kernel_number])
plt.show()