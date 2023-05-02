# uses the google vgg16 net for image classification
# on the imagenet challenge

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from matplotlib import pyplot as plt

img = load_img('..\data\dog.jpg', target_size=(224, 224))
plt.imshow(img)
plt.show()

# process image for vgg16 use
img = img_to_array(img)
img = img.reshape((1, 224, 224, 3))
img = preprocess_input(img)
print(img)

# image augmentation
gen = ImageDataGenerator()
train_generator =

# prediction
model = VGG16()
ypred = model.predict(img)
ypred = decode_predictions(ypred)
