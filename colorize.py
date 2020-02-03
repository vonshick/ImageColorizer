
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from PIL import ImageFile
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 

def load_image(file):
    image = load_img(file)
    # plt.imshow(image)
    # plt.show()
    image = img_to_array(image)
    image = np.array(image, dtype=float)

    return image


def get_input_and_output(image):
    gray = rgb2lab(1.0/255*image)[:,:,0]
    color = rgb2lab(1.0/255*image)[:,:,1:]
    
    # make color to take values from range (-1; 1)
    color /= 128 
    
    gray = gray.reshape(1, 400, 400, 1)
    color = color.reshape(1, 400, 400, 2)

    return gray, color


def define_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    return model


def save_output(gray, colorized):
    output = np.zeros((400, 400, 3))
    output[:,:,0] = gray[0][:,:,0]
    output[:,:,1:] = colorized[0]
    imsave("img_result.png", lab2rgb(output))
    imsave("img_gray_version.png", rgb2gray(lab2rgb(output)))


def train_model(batch_size, epochs):
    image = load_image('input/all/abomasnow.png')
    # image = cv2.imread('input/all/abra.png')

    gray, color = get_input_and_output(image)
    model = define_model()
    model.compile(optimizer='rmsprop',loss='mse')
    model.fit(x=gray, 
        y=color,
        batch_size=1,
        epochs=epochs)

    print(model.evaluate(gray, color, batch_size = 1))

    colorized = model.predict(gray)
    colorized *= 128

    save_output(gray, colorized)


if __name__ == "__main__":
    train_model(batch_size = 1, epochs = 100)