# -*- coding: utf-8 -*-
"""

Very minimal architecture similar to Keras basic autoencoder 

Mike Fu
"""

import keras
import tensorflow as tf
import keras.backend

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate
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

class BruceNet(SuperModel):
    def __init__(self, input_size=(128, 128, 3)):

        input_img = Input(shape=input_size)

        # 1 by 1 conv layer
        x = Conv2D(6, (1, 1), activation='selu', padding='same')(input_img)

        # encoder layer
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)  # size = (32, 32)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)  # size = (16, 16)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)  # size = (8, 8)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        encoded = Conv2D(128, (3, 3), activation='selu', padding='same', name='encoded')(x)

        # oversight
        x = MaxPooling2D((2,2), padding='same')(encoded)  # size = (4, 4)
        x = MaxPooling2D((2,2), padding='same')(x)  # size = (1, 1)
        x = Conv2D(128, (4, 4), activation='selu', padding='valid')(x)
        oversight = UpSampling2D((8,8), name='oversight')(x) # (8, 8, 128)

        # imap 1st deconv
        x = MaxPooling2D((2,2), padding='same')(encoded)
        imap_1 = Conv2D(128, (3, 3), activation='selu', padding='same')(x)

        # mmap 1st deconv
        x = MaxPooling2D((2,2), padding='same')(encoded)
        mmap_1 = Conv2D(128, (3, 3), activation='selu', padding='same')(x)

        # concatenate the three layers together
        concat_1 = Concatenate()([oversight, imap_1, mmap_1])
        # print(concat_1.shape)  # should be 128 * 3

        # deconv structure for imap part 1
        x = Conv2D(256, (3, 3), activation='selu', padding='same')(concat_1)
        x = Conv2D(256, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        imap_2 = UpSampling2D((2, 2))(x)

        # same deconv structure for the mmap
        x = Conv2D(256, (3, 3), activation='selu', padding='same')(concat_1)
        x = Conv2D(256, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', padding='same')(x)
        mmap_2 = UpSampling2D((2, 2))(x)

        # cross communication one more time
        concat_2 = Concatenate()([imap_2, mmap_2])

        # last conv stack for imap
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(concat_2)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        decoded_imap = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded_imap')(x)
        # last conv stack for mmap
        x = Conv2D(16, (3, 3), activation='selu', padding='same')(concat_2)
        x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
        decoded_mmap = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded_mmap')(x)


        self.model = Model(input_img, [decoded_imap, decoded_mmap])
        self.model.compile(optimizer='adam', loss=self.custom_loss())
        
        
    def custom_loss(self):

        def decoded_imap(true_img, pred_img):

            imap_diff = K.mean(K.square((0.5 * true_img) - pred_img))
            return imap_diff

        def decoded_mmap(true_img, pred_img):

            mmap_diff = K.mean(K.square(true_img - pred_img))
            return mmap_diff

        return {'decoded_imap': decoded_imap, 'decoded_mmap': decoded_mmap}
        
