# -*- coding: utf-8 -*-
"""

Very minimal architecture similar to Keras basic autoencoder 

Mike Fu


NOTE Put you have changed here:

TestJank2: Changed selu to relu


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

class TestJankNet(SuperModel):
    def __init__(self, input_size=(128, 128, 3)):

        print('\n\n\n ***** Testing Janknet test 2 **** \n\n\n')

        input_img = Input(shape=input_size)

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
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded_imap = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded_imap')(x)

        # same deconv structure for the mmap
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(encoded)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded_mmap = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded_mmap')(x)


        self.model = Model(input_img, [decoded_imap, decoded_mmap])
        self.model.compile(optimizer='adam', loss=self.custom_loss)
        
        
    def custom_loss(self, true_img, pred_img):

        print('shape', true_img[2].shape)

        imap_diff = K.mean(K.square((0.5 * true_img[0]) - pred_img[0]))
        mmap_diff = K.mean(K.square(true_img[1] - pred_img[1]))
        return imap_diff + mmap_diff


    # def __str__(self):
    #     return self.model.summary()
    
    # def train(self, len_data, batch_size, num_epochs, gen, callbacks_list=[]):
    #     '''
    #         call this to train the network
    #         gen - a generator function to pass into model.fit_generator()
    #     '''
    #     return self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=1)

    # def evaluate(self, len_data, batch_size, gen, callbacks_list=[]):
    #     '''
    #         call this to run keras evaluate_generator on the network
    #     '''
    #     return self.model.evaluate_generator(gen, steps=len_data / batch_size, verbose=1)

        
        




