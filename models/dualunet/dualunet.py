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

class DualUNet(SuperModel):

    '''
        Unet model
    '''

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


        # expansion (decoding) stage for imap
        ct50i = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c41)
        sk50i = Concatenate()([ct50i, c31])
        c50i = Conv2D(128, (3, 3), activation='selu', padding='same')(sk50i)
        c51i = Conv2D(128, (3, 3), activation='selu', padding='same')(c50i)

        # 
        ct60i =  Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c51i)
        sk60i = Concatenate()([ct60i, c21])
        c60i = Conv2D(64, (3, 3), activation='selu', padding='same')(sk60i)
        c61i = Conv2D(64, (3, 3), activation='selu', padding='same')(c60i)

        # 
        ct70i = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c61i)
        sk70i = Concatenate()([ct70i, c11])
        c70i = Conv2D(32, (3, 3), activation='selu', padding='same')(sk70i)
        c71i = Conv2D(32, (3, 3), activation='selu', padding='same')(c70i)

        #
        ct80i = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c71i)
        sk80i = Concatenate()([ct80i, c01])
        c80i = Conv2D(16, (3, 3), activation='selu', padding='same')(sk80i)
        c81i = Conv2D(16, (3, 3), activation='selu', padding='same')(c80i)



        # expansion decoding for mmap
        ct50m = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c41)
        sk50m = Concatenate()([ct50m, c31])
        c50m = Conv2D(128, (3, 3), activation='selu', padding='same')(sk50m)
        c51m = Conv2D(128, (3, 3), activation='selu', padding='same')(c50m)

        # 
        ct60m =  Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c51m)
        sk60m = Concatenate()([ct60m, c21])
        c60m = Conv2D(64, (3, 3), activation='selu', padding='same')(sk60m)
        c61m = Conv2D(64, (3, 3), activation='selu', padding='same')(c60m)

        # 
        ct70m = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c61m)
        sk70m = Concatenate()([ct70m, c11])
        c70m = Conv2D(32, (3, 3), activation='selu', padding='same')(sk70m)
        c71m = Conv2D(32, (3, 3), activation='selu', padding='same')(c70m)

        #
        ct80m = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c71m)
        sk80m = Concatenate()([ct80m, c01])
        c80m = Conv2D(16, (3, 3), activation='selu', padding='same')(sk80m)
        c81m = Conv2D(16, (3, 3), activation='selu', padding='same')(c80m)

        # conv 1 x 1 cross channel communication
        output_mmap = Conv2D(3, (1, 1), activation='sigmoid', name='decoded_mmap')(c81m)



        # conv 1 x 1 cross channel communication
        output_imap = Conv2D(3, (1, 1), activation='sigmoid', name='decoded_imap')(c81i)



        self.model = Model(inputs=[input_img], outputs=[output_imap, output_mmap])

        # compile the model with a loss function
        self.model.compile(optimizer='adam', loss=self.custom_loss(), loss_weights={'decoded_imap': 0.7, 'decoded_mmap': 0.3})


    def custom_loss(self):
        # function names should match with the names of the corresponding output layers
        def decoded_imap(true_img, pred_img):
            return K.mean(K.square(true_img * 0.5 - pred_img))

        def decoded_mmap(true_img, pred_img):
            return K.mean(K.square(true_img - pred_img))

        return {'decoded_imap': decoded_imap, 'decoded_mmap': decoded_mmap}