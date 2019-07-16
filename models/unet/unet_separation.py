# -*- coding: utf-8 -*-

# Allen Ma 
# unet initial experiments with just generating half the imap

import sys
sys.path.insert(0, '../')

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

from supermodel import superModel

class UNet(superModel):


    def imap_only_loss(self, true_img, pred_img):
        return K.mean(K.square(true_img * 0.5 - pred_img))

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
        output_img = Conv2D(3, (1, 1), activation='sigmoid')(c81)

        self.model = Model(inputs=[input_img], outputs=[output_img])

        print(self.model.summary())

        # compile the model with a loss function
        self.model.compile(optimizer='adam', loss=self.imap_only_loss, metrics=['mse'])

    def train(self, len_data, batch_size, num_epochs, gen, callbacks_list=[]):
        '''
            call this to train the network
            gen - a generator function to pass into model.fit_generator()
        '''
        return self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks_list)


def main():

    # NOTE: hardcoding is as follows
    # imap_npy has 4800 images
    # mmap_npy has 1200 images
    # so by design, we multiply mmap * 4
    # batch size 64 bc 4800 / 64 = 75 which is cleanly divisible
    # file paths are hardcoded for the linux dwarves
    # - Allen 2 July

    unet = UNet()

    # hardcoded path names
    # in data_gen you apparently have to specify final
    path_imap = "/media/yma21/gilmore/intrinsic-images/data/imap_npy/final"
    path_mmap = "/media/yma21/gilmore/intrinsic-images/data/matmap_npy/"

    BATCH_SIZE = 64
    LEN_DATA = 4800
    EPOCHS = 50

    # checkpoint
    filepath="./weights-unet-{epoch:02d}-{loss:.2f}.hdf5"
    # save the minimum loss
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]
    # Fit the model
    unet.train(LEN_DATA, BATCH_SIZE, EPOCHS, data_gen.generator(path_imap, path_mmap), callbacks_list)


if __name__ == "__main__":
    main()




