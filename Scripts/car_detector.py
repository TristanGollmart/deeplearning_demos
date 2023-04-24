import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input
from keras.models import Model

# data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train == 1 # car class is 1
y_test = y_test == 1

X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# ---------------- Generate New Data--------------------------------------------------------------
gen = ImageDataGenerator(width_shift_range=3, height_shift_range=3, zoom_range=0.1, horizontal_flip=True)
# for batch in gen.flow(X_train, y_train, shuffle=False):
#     print(batch[0].shape)
#     first_image = batch[0][0]
#     plt.imshow(first_image)
#     plt.show()
#     break
# ---------------------------------------------------------

# plt.imshow(X_train[1])
# plt.show()
# model
input = Input(shape=(32, 32, 3))
conv1 = Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(32, 32, 3), activation='relu')(input)
maxpool1 = MaxPool2D(pool_size=(2, 2))(conv1)
maxpool1 = Dropout(0.25)(maxpool1)

conv2 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(maxpool1)
maxpool2 = MaxPool2D(pool_size=(2, 2))(conv2)
maxpool2 = Dropout(0.25)(maxpool2)

conv3 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(maxpool2)
maxpool3 = MaxPool2D(pool_size=(2, 2))(conv3)
maxpool3 = Dropout(0.25)(maxpool3)

flat = Flatten()(maxpool3)
dense1 = Dense(128, activation='relu')(flat)
output = Dense(1, activation='sigmoid')(dense1)

model = Model(input, output)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# ---------FITTING PROCESS-------------
# history = model.fit_generator(gen.flow(X_train, y_train, batch_size=128), epochs=10)
# model.evaluate(X_test, y_test)
# model.save(r'..\models\cnn_car_detector')

# ----- Load saved model without retraining
model_loaded = tf.keras.models.load_model(r'..\models\cnn_car_detector')
print(f'loaded model performance {model_loaded.evaluate(X_test, y_test)}')





