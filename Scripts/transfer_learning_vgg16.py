# uses the google vgg16 net for image classification
# on the imagenet challenge

from keras.utils import load_img

img = load_img('..\data\dog.jpg')
print(img)
