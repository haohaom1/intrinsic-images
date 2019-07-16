'''
Super class for models in training

'''

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


class superModel():

    # should override the init for each custom model
    def __init__(self, input_size=(128, 128, 3)):
        self.input_img = Input(shape=input_size)
        self.model = Model(self.input_img, self.input_img)

    def __str__(self):
        return self.model.summary()

    def train(self, len_data, batch_size, num_epochs, gen, callbacks_list=[]):
        '''
            call this to train the network
            gen - a generator function to pass into model.fit_generator()
        '''
        return self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=1)

    def evaluate(self, len_data, batch_size, gen, callbacks_list=[]):
        '''
            call this to run keras evaluate_generator on the network
        '''
        return self.model.evaluate_generator(gen, steps=len_data / batch_size, verbose=1)