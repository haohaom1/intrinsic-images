# -*- coding: utf-8 -*-
"""

Janknet, but more powerful
Namely, more convolutional layers to the encoding stage

Allen Ma
"""

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

class StrongerJanknet(SuperModel):
    def __init__(self, input_size=(128, 128, 3)):

        input_img = Input(shape=input_size)

        # encoder layer
        x = Conv2D(6, (1, 1), activation='selu', padding='same')(input_img)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x) # 64 x 64 
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x) # 32 x 32
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x) # 16 x 16
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x) # 8 x 8
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x) # 4 x 4
        x = Conv2D(256, (3, 3), activation='selu', padding='same')(x)
        encoded = Conv2D(256, (3, 3), activation='sigmoid', padding='same')(x)
        # ensure everything is between 0 and 1

        x = Conv2D(128, (3, 3), activation='selu', padding='same')(encoded)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded_imap = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded_imap')(x)

        x = Conv2D(128, (3, 3), activation='selu', padding='same')(encoded)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded_mmap = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded_mmap')(x)


        self.model = Model(input_img, [decoded_imap, decoded_mmap])
        self.model.compile(optimizer='adam', loss=self.custom_loss)
        
        
    def custom_loss(self, true_img, pred_img):
        imap_diff = K.mean(K.square((0.5 * true_img[0]) - pred_img[0]))
        mmap_diff = K.mean(K.square(true_img[1] - pred_img[1]))
        return imap_diff + mmap_diff


    
        
        




