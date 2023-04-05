import keras
import tensorflow as tf
import gzip
import numpy as np
from tensorflow.keras.models import Model
from keras.layers import Input,Flatten, Dropout, Dense, MaxPool2D, Conv2D
from keras.utils import to_categorical

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
y_train = open_images("../data/fashion/train-images-idx1-ubyte.gz")

X_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")
y_test = open_images("../data/fashion/t10k-images-idx3-ubyte.gz")

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model using keras functional API
input = tf.keras.Input(shape=(28, 28))
conv1 = Conv2D(10, kernel_size = (3, 3), activation='sigmoid', input_shape=(784,))(input)
output = Dense(10, activation='sigmoid')

model = Model(input, output)