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

import json

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
        return self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=2, callbacks=callbacks_list)
        
        
def main(argv):

    janknet = JankNet()

    # Using argv for path names
    path_imap = argv[1]
    path_mmap = argv[2]
    history_path = argv[3]

    BATCH_SIZE = 64
    NUM_MMAPS_PER_IMAP = 5
    LEN_DATA = min(len(os.listdir(path_imap)), len(os.listdir(path_mmap)) * NUM_MMAPS_PER_IMAP)
    EPOCHS = 50

    # checkpoint
    filepath="./weights-janknet-{epoch:02d}-{loss:.2f}.hdf5"
    # save the minimum loss
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=False)
    callbacks_list = [checkpoint]
    # Fit the model
    history_obj = janknet.train(LEN_DATA, BATCH_SIZE, EPOCHS, data_gen.generator(path_imap, path_mmap, num_mmaps_per_imap=NUM_MMAPS_PER_IMAP), callbacks_list)
    # save the history object to a pickle file
    json.dump(history_obj.history, open(history_path, "w"))


if __name__ == "__main__":
    main(sys.argv)



