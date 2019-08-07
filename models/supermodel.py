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

from keras.utils import plot_model

import os


class SuperModel():

    # should override the init for each custom model
    def __init__(self, input_size=(128, 128, 3)):
        self.input_img = Input(shape=input_size)
        self.model = Model(self.input_img, self.input_img)

    def __str__(self):
        return self.model.summary()

    def train(self, len_data, batch_size, num_epochs, gen, validation_gen=None, validation_len_data=None, callbacks=[]):
        '''
            call this to train the network
            gen - a generator function to pass into model.fit_generator()
        '''
        if not validation_gen:
            return self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=1, validation_data=None, validation_steps= None, callbacks=callbacks)
        else:
            return self.model.fit_generator(gen, steps_per_epoch= len_data / batch_size, epochs=num_epochs, verbose=1, validation_data=validation_gen, validation_steps= validation_len_data / batch_size, callbacks=callbacks)


    def evaluate(self, len_data, batch_size, gen, callbacks_list=[]):
        '''
            call this to run keras evaluate_generator on the network
        '''
        return self.model.evaluate_generator(gen, steps=len_data / batch_size, verbose=1)

    def load_weights(self, path):
        self.model.load_weights(path)

    # default to SSD
    def custom_loss(self, true_img, pred_img):
       return K.mean(K.square(true_img - pred_img))

    # saves the model architecture as png
    def save_model_architecture(self, fname, path=None):
        if not fname.endswith('.png'):
            fname += '.png'
        if path:
            fname = os.path.join(path, fname)

        # save architecture if it doesn't exist already
        if not os.path.isfile(fname): 
            plot_model(self.model, to_file=fname)