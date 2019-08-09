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
from models.dualunet.dualunet import DualUNet

class PmaxDualUNet(DualUNet):

    '''
        Dual Unet model with pixelmax loss function
    '''

    def __init__(self, input_size=(128, 128, 3)):
        self.input_img = Input(shape=input_size)
        super.__init__(input_size=input_size)

    def custom_loss(self):
        # function names should match with the names of the corresponding output layers
        # uses pixel max loss as a scale

        # finds max intensity, assumes color channel is last
        idx = K.argmax(K.flatten(K.mean(self.input_img, axis=-1)))
        pmax = K.flatten(K.mean(self.input_img, axis=-1))[idx]         # max intensity, a number

        def decoded_imap(true_img, pred_img):
            scale = pmax / K.flatten(K.mean(true_img, axis=-1))[idx]     
             
            return K.mean(K.square((true_img / scale) - pred_img))

        def decoded_mmap(true_img, pred_img):
            scale = pmax / K.flatten(K.mean(true_img, axis=-1))[idx]    

            return K.mean(K.square((true_img * scale) - pred_img))

        return {'decoded_imap': decoded_imap, 'decoded_mmap': decoded_mmap}