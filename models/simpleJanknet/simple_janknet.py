# -*- coding: utf-8 -*-
"""

Very minimal architecture, inspired by Bruce Maxwell's autoencoder for MNIST
Might need to add more channels / layers
Task: produce a imap that is half of the one fed in

Mike Fu

"""
import os

import keras
import tensorflow as tf
import keras.backend

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
from keras.models import load_model

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.losses import mse

from models.supermodel import SuperModel

# true image is the illumination map that was used to construct the input image
# pred image is the generated illumination map * 0.5 


# input_img = Input(shape=(128, 128, 3))  # adapt this if using `channels_first` image data format

class SimpleJankNet(SuperModel):
    def __init__(self, input_size=(128, 128, 3)):

        # define input
        input_img = Input(shape=input_size)

        # encoder layer
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        encoded = Conv2D(2, (1, 1), activation='sigmoid', padding='same')(x)
        # ensure everything is between 0 and 1

        # at this point the representation is (16, 16, 32) i.e. 2048-dimensional

        x = Conv2D(8, (3, 3), activation='selu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded_imap = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(input_img, decoded_imap)
        self.model.compile(optimizer='adam', loss=self.custom_loss)
        
        
    def custom_loss(self, true_img, pred_img):
       return K.mean(K.square(0.5 * true_img - pred_img))
        
