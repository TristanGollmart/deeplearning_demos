# uses the google vgg16 net for image classification
# on the imagenet challenge
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

TRAIN_CUSTOM_MODEL = False
TRAIN_VGG_MODEL = True

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

X_train, X_test, y_train, y_test = train_test_split(X_customNet, y, test_size=0.2, random_state=42)

# custom cnn

if TRAIN_CUSTOM_MODEL:
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
    X_train, y_train = shuffle(X_train, y_train) # to make sure cats and dogs are mixed appropriatly since validation set always cuts of last data
    history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

    model.save(os.path.join("..", "models", "catDogClassifier"))
else:
    model = load_model(os.path.join("..", "models", "catDogClassifier"))
test_result = model.evaluate(X_test, y_test)


# ---------------------
# ----- vgg16 model ---
# ---------------------


X = preprocess_input(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if TRAIN_VGG_MODEL:
    model_vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))
    model_vgg16.trainable = False

    # for speed: do forward pass of vgg only once before training, since weights will not change
    X_vgg_output_train = model_vgg16.predict(X_train, verbose=1)
    # for validation set: cuts of last samples, otherwise would only be cats
    X_vgg_output_train, y_train = shuffle(X_vgg_output_train, y_train)

    input = Input(shape=(7, 7, 512)) # output shape of the vgg-head
    vgg16 = Flatten()(input)
    vgg16 = Dense(4096, activation='relu')(vgg16)
    vgg16 = Dense(1024, activation='relu')(vgg16)
    output = Dense(1, activation='sigmoid')(vgg16)
    model = Model(input, output)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=['accuracy'])
    model.fit(X_vgg_output_train, y_train, batch_size=32, epochs=30, validation_split=0.2)
    model.save(os.path.join("..", "models", "vgg16TransferLearning"))
else:
    model_vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))
    model = load_model(os.path.join("..", "models", "vgg16TransferLearning"))

test_result = model.evaluate(model_vgg16.predict(X_test), y_test)
ypred = model.predict(model_vgg16.predict(X_test[0].reshape(1, 224, 224, 3)))
if ypred[0][0] < 0.5:
    print("this image is not a dog")
else:
    print("this image is a dog")

# process test image for vgg16 use
img = load_img(r'..\\data\\dog.jpg', target_size=(224, 224))
plt.imshow(img)
plt.show()
img = img_to_array(img)
img = img.reshape((1, 224, 224, 3))
img = preprocess_input(img)
print(img)

ypred = model.predict(model_vgg16.predict(img))

if ypred[0][0] < 0.5:
    print("this image is not a dog")
else:
    print("this image is a dog")

# on a test image not from that dataset

