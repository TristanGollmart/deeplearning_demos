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
#img = draw_rectangle(draw, [50, 50], [150, 150])

# plt.imshow(img)
# plt.show()


# load fitted model from "car_detector"
model = tf.keras.models.load_model(r"..\models\cnn_car_detector")

# sliding window over image to find car
# cropping
window_size = 100
step_size = 20
nsteps_x = int(img.size[0] / step_size) + 1
nsteps_y = int(img.size[1] / step_size) + 1

y_data = []
for i in range(nsteps_x):
    for j in range(nsteps_y):
        img_cropped = img.crop((i * step_size, j * step_size,
                                i * step_size + window_size, j * step_size + window_size))\
                                .resize((32, 32), resample=Image.Resampling.BICUBIC)

        data = np.asarray(img_cropped).astype(np.float32) / 255.
        data = data.reshape((1, 32, 32, 3))
        y_data.append(model.predict(data)[0][0])

assert len(y_data) == nsteps_x * nsteps_y, "length of prediction array is not as expected"
y_data = np.reshape(y_data, (nsteps_x, nsteps_y))

# (i_opt, j_opt) = np.unravel_index(y_data.argmax(), y_data.shape)
# point_corner_opt = [i_opt * step_size, j_opt * step_size]
# point_corner_opt2 = [i_opt * step_size + window_size, j_opt * step_size + window_size]
# draw = ImageDraw.Draw(img)
# img_signed = draw_rectangle(draw, point_corner_opt, point_corner_opt2)

points = []
points2 = []
img_signed = img.copy()
for ix in range(y_data.shape[0]):
    for iy in range(y_data.shape[1]):
        if y_data[ix, iy] >= 0.4:
            points.append([ix * step_size, iy*step_size])
            points2.append([ix * step_size + window_size, iy*step_size + window_size])
            draw = ImageDraw.Draw(img_signed)
            img_signed = draw_rectangle(draw, points[-1], points2[-1])

plt.imshow(img_signed)
plt.show()

print('finished')
