# -*- coding: utf-8 -*-
"""

Very minimal architecture similar to Keras basic autoencoder 

Allen Ma
"""

import keras
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import keras.backend

import numpy as np
# %matplotlib inline
import random

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
from keras.models import load_model

from keras.callbacks import TensorBoard
import os, os.path
from keras.callbacks import ModelCheckpoint
from keras.losses import mse

# true image is the illumination map that was used to construct the input image
# pred image is the generated illumination map * 0.5 
def custom_loss(true_img, pred_img):
   return K.mean(K.square(true_img * 0.5 - pred_img))

input_img = Input(shape=(128, 128, 3))  # adapt this if using `channels_first` image data format

# encoder layer
x = Conv2D(8, (3, 3), activation='selu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
x = Dropout(0.2,name='drop1')(x)
x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
x = Conv2D(32, (1, 1), activation='sigmoid', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# ensure everything is between 0 and 1

# at this point the representation is (16, 16, 32) i.e. 2048-dimensional

x = Conv2D(32, (3, 3), activation='selu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='selu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded_imap = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


separator = Model(input_img, decoded_imap)
separator.compile(optimizer='adam', loss=custom_loss, metrics=['mse'])



separator.summary()

# checkpoint
filepath="./weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
# save the minimum loss
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False)
callbacks_list = [checkpoint]
# Fit the model

separator.fit(x_train, [vert_train, horiz_train],
                epochs=30,
                batch_size=84,
                shuffle=True,
                validation_split=0.1,
                callbacks=callbacks_list)

