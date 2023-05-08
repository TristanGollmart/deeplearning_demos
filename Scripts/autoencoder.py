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
y_val = to_categorical(y_val)