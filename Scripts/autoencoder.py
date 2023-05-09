import keras
import tensorflow as tf
import gzip
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense, MaxPool2D, Conv2D, UpSampling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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


Y_train = open_images("../data/mnist/train-images-idx3-ubyte.gz")
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
Y_test = open_images("../data/mnist/t10k-images-idx3-ubyte.gz")

# plt.imshow(X_train[0])
# plt.show()

# target: clean images, inputs: noisy images
Y_train = Y_train.astype(np.float32) / 255.
Y_test = Y_test.astype(np.float32) / 255.
X_train = Y_train + np.random.normal(0, 0.2, size=Y_train.shape)
X_test = Y_test + np.random.normal(0, 0.2, size=Y_test.shape)

input = Input(shape=(28, 28, 1))
encode = Conv2D(filters=1, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu')(input)
encode = MaxPool2D(pool_size=(2, 2))(encode)

decode = Conv2D(5, kernel_size=(3, 3), padding='same', activation='relu')(encode)
decode = UpSampling2D(size=(2, 2))(decode)
decode = Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid')(decode)

model = Model(input, decode)


model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train.reshape(-1, 28, 28, 1), Y_train.reshape(-1, 28, 28, 1), epochs=10, batch_size=32)
model.save(r'..\models\mnist_autoencoder')

model_encoder = Model(input, encode)

# use auto encoder to remove noise from a test image

image_test = X_train[0]
image_restored = model.predict(image_test.reshape(1, 28, 28, 1))
image_encoded = model_encoder.predict(image_test.reshape(1, 28, 28, 1))

plt.imshow(image_test)
plt.imshow(image_restored.reshape(28, 28))
plt.imshow(image_encoded.reshape(14, 14))


print(history.history)



print("finished")