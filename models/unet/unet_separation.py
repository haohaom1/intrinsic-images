# -*- coding: utf-8 -*-

# Allen Ma 
# unet initial experiments with just generating half the imap

import keras
import tensorflow as tf

import keras.backend

import numpy as np
# %matplotlib inline
import random

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, Concatenate
from keras.models import Model
from keras import backend as K
from keras.models import load_model

from keras.callbacks import TensorBoard
import os, os.path
from keras.callbacks import ModelCheckpoint
from keras.losses import mse

import data_gen

from models.supermodel import SuperModel

class UNet(SuperModel):

    '''
        Unet model
    '''

    def custom_loss(self):
        # function names should match with the names of the corresponding output layers
        def decoded_imap(true_img, pred_img):
            return K.mean(K.square(true_img * 0.5 - pred_img))

        return {'decoded_imap': decoded_imap}

    def __init__(self, input_size=(128, 128, 3)):
        input_img = Input(shape=input_size)

        # contraction stage

        # 16, 3, 3 followed by maxpool of stride 2
        c00 = Conv2D(16, (3, 3), activation='selu', padding='same')(input_img)
        c01 = Conv2D(16, (3, 3), activation='selu', padding='same')(c00)
        m00 = MaxPooling2D((2, 2), padding='same')(c01)
        # 32, 3, 3 followed by maxpool of stride 2
        c10 = Conv2D(32, (3, 3), activation='selu', padding='same')(m00)
        c11 = Conv2D(32, (3, 3), activation='selu', padding='same')(c10)
        m10 = MaxPooling2D((2, 2), padding='same')(c11)
        # 64, 3, 3, followed by maxpool of stride 2
        c20 = Conv2D(64, (3, 3), activation='selu', padding='same')(m10)
        c21 = Conv2D(64, (3, 3), activation='selu', padding='same')(c20)
        m20 = MaxPooling2D((2, 2), padding='same')(c21)
        # 128, 3, 3, followed by maxpool of stride 2
        c30 = Conv2D(128, (3, 3), activation='selu', padding='same')(m20)
        c31 = Conv2D(128, (3, 3), activation='selu', padding='same')(c30)
        m30 = MaxPooling2D((2, 2), padding='same')(c31)

        # 256, 3, 3, followed by maxpool of stride 2
        c40 = Conv2D(256, (3, 3), activation='selu', padding='same')(m30)
        c41 = Conv2D(256, (3, 3), activation='selu', padding='same')(c40)


        # expansion (decoding) stage

        # 
        ct50 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c41)
        sk50 = Concatenate()([ct50, c31])
        c50 = Conv2D(128, (3, 3), activation='selu', padding='same')(sk50)
        c51 = Conv2D(128, (3, 3), activation='selu', padding='same')(c50)

        # 
        ct60 =  Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c51)
        sk60 = Concatenate()([ct60, c21])
        c60 = Conv2D(64, (3, 3), activation='selu', padding='same')(sk60)
        c61 = Conv2D(64, (3, 3), activation='selu', padding='same')(c60)

        # 
        ct70 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c61)
        sk70 = Concatenate()([ct70, c11])
        c70 = Conv2D(32, (3, 3), activation='selu', padding='same')(sk70)
        c71 = Conv2D(32, (3, 3), activation='selu', padding='same')(c70)

        #
        ct80 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c71)
        sk80 = Concatenate()([ct80, c01])
        c80 = Conv2D(16, (3, 3), activation='selu', padding='same')(sk80)
        c81 = Conv2D(16, (3, 3), activation='selu', padding='same')(c80)

        # conv 1 x 1 cross channel communication
        output_img = Conv2D(3, (1, 1), activation='sigmoid', name='decoded_imap')(c81)

        self.model = Model(inputs=[input_img], outputs=[output_img])

        # compile the model with a loss function
        self.model.compile(optimizer='adam', loss=self.custom_loss())
