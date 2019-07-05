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


input_img = Input(shape=(128, 128, 3))  # adapt this if using `channels_first` image data format

class JankNet():
    def __init__(self):
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


        self.model = Model(input_img, decoded_imap)
        self.model.compile(optimizer='adam', loss=self.imap_only_loss, metrics=['mse'])
        
        
    def imap_only_loss(self, true_img, pred_img):
       return K.mean(K.square(true_img * 0.5 - pred_img))


    def __str__(self):
        return self.model.summary()
    
        def train(self, len_data, batch_size, num_epochs, gen, callbacks_list=[]):
        '''
            call this to train the network
            gen - a generator function to pass into model.fit_generator()
        '''
        self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=2, callbacks=callbacks_list)
        
        
def main():

    # NOTE: hardcoding is as follows
    # imap_npy has 4800 images
    # mmap_npy has 1200 images
    # so by design, we multiply mmap * 4
    # batch size 64 bc 4800 / 64 = 75 which is cleanly divisible
    # file paths are hardcoded for the linux dwarves
    # - Allen 2 July

    janknet = JankNet()

    # hardcoded path names
    # in data_gen you apparently have to specify final
    path_imap = "/media/yma21/gilmore/intrinsic-images/data/imap_npy/final"
    path_mmap = "/media/yma21/gilmore/intrinsic-images/data/matmap_npy/"

    BATCH_SIZE = 64
    LEN_DATA = 4800
    EPOCHS = 50

    # checkpoint
    filepath="./weights-janknet-{epoch:02d}-{loss:.2f}.hdf5"
    # save the minimum loss
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=False)
    callbacks_list = [checkpoint]
    # Fit the model
    janknet.train(LEN_DATA, BATCH_SIZE, EPOCHS, data_gen.generator(path_imap, path_mmap), callbacks_list)


if __name__ == "__main__":
    main()



