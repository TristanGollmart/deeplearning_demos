# uses the google vgg16 net for image classification
# on the imagenet challenge
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# process test image for vgg16 use
# img = load_img('..\data\dog.jpg', target_size=(224, 224))
# plt.imshow(img)
# plt.show()
# img = img_to_array(img)
# img = img.reshape((1, 224, 224, 3))
# img = preprocess_input(img)
# print(img)

# read_images
def read_images(path):
    # read in and process all jpg files
    files = os.listdir(path)
    jpg_files = [file for file in files if file[-4:] == '.jpg']
    images = []
    for file in tqdm(jpg_files[:1000]):
        try:
            image = Image.open(os.path.join(path, file))
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            image = image.convert(mode="RGB")
            image = np.asarray(image)
            images.append(image)
        except OSError:
            pass
    return images


cats = np.asarray(read_images(os.path.join("..", "data", "PetImages", "Cat")))
dogs = np.asarray(read_images(os.path.join("..", "data", "PetImages", "Dog")))
X = np.concatenate([dogs, cats])
X_customNet = X.astype(np.float32) / 255.
y_cats = np.zeros(len(cats))
y_dogs = np.ones(len(dogs))
y = np.concatenate([y_dogs, y_cats]).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# custom cnn
input = Input(shape=(224, 224, 3))
conv1 = Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3), activation='relu')(input)
conv1 = Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3), activation='relu')(conv1)
maxpool1 = MaxPool2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool1)
conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
maxpool2 = MaxPool2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(maxpool2)
conv3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
maxpool3 = MaxPool2D(pool_size=(2, 2))(conv3)
maxpool3 = Dropout(0.25)(maxpool3)

flat = Flatten()(maxpool3)
dense1 = Dense(1024, activation='relu')(flat)
dense1 = Dropout(0.25)(dense1)
dense1 = Dense(128, activation='relu')(dense1)
dense1 = Dropout(0.25)(dense1)
output = Dense(1, activation='sigmoid')(dense1)

model = Model(input, output)

model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)
test_result = model.evaluate(X_test, y_test)
model.save(os.path.join("..", "models", "catDogClassifier"))

# vgg16 model
X = preprocess_input(X)

model_vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))
model_vgg16.trainable=False

input = Input(shape=(224, 224, 3))
vgg16 = model_vgg16(input)
vgg16 = Flatten()(vgg16)
vgg16 = Dense(4096, activation='relu')(vgg16)
vgg16 = Dense(1024, activation='relu')(vgg16)
output = Dense(1, activation='sigmoid')(vgg16)
model = Model(input, output)
model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=30)

ypred = model.predict(img)
ypred = decode_predictions(ypred)

