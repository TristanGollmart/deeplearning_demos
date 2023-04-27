import PIL.Image
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input
from keras.models import Model
from PIL import Image, ImageDraw

def draw_rectangle(draw, p1, p2):
    draw.line([(p1[0], p1[1]),
               (p1[0], p2[1]),
               (p2[0], p2[1]),
               (p2[0], p1[1]),
               (p1[0], p1[1])], "yellow", 5)
    return img

img = Image.open(r'..\data\bild1.jpg')
img = img.resize((round(img.size[0]/10), round(img.size[1]/10)), resample=Image.Resampling.LANCZOS)
# img.rotate(90, expand=True)

#draw = ImageDraw.Draw(img)
draw = ImageDraw.Draw(img)
img = draw_rectangle(draw, [50, 50], [150, 150])

plt.imshow(img)
plt.show()

print('finished')
