import keras
import tensorflow as tf
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import keras.backend as K

import numpy as np


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.utils import plot_model

# from models.supermodel import SuperModel

class CycleGAN():

    def __init__(self, input_size=(128, 128, 3)):

        # network 1, image 1
        input_img_1 = Input(shape=input_size, name='input_1')  
        encoded_1 = self.encoder(input_img_1, name='encoded_1')
        reflectance_1 = self.decoder(encoded_1, name='reflectance_1')
        illumination_1 = self.decoder(encoded_1, name='illumination_1')

        # network 1, image 2
        input_img_2 = Input(shape=input_size, name='input_2')  
        encoded_2 = self.encoder(input_img_2, name='encoded_2')
        reflectance_2 = self.decoder(encoded_2, name='reflectance_2')
        illumination_2 = self.decoder(encoded_2, name='illumination_2')

        # network 2, image 1'
        image_1_prime = keras.layers.Multiply(name='image_1_p')([reflectance_1, illumination_2])
        encoded_1_prime = self.encoder(image_1_prime, name='encoded_1_p')
        reflectance_1_prime = self.decoder(encoded_1_prime, name='reflectance_1_p')
        illumination_1_prime = self.decoder(encoded_1_prime, name='illumination_1_p')
        
        # network 2, image 2'
        image_2_prime = keras.layers.Multiply(name='image_2_p')([reflectance_2, illumination_1])
        encoded_2_prime = self.encoder(image_2_prime, name='encoded_2_p')
        reflectance_2_prime = self.decoder(encoded_2_prime, name='reflectance_2_p')
        illumination_2_prime = self.decoder(encoded_2_prime, name='illumination_2_p')

        self.model = Model([input_img_1, input_img_2], [reflectance_1_prime, illumination_1_prime, reflectance_2_prime, illumination_2_prime])
        self.model.compile(optimizer='adam', loss=self.custom_loss, metrics=['mse'])


    def custom_loss(self, y_true, y_pred):

        # mmap1, imap2, mmap2, imap1 = tuple(y_true)
        r1p = y_pred[0]
        l1p = y_pred[1]
        r2p = y_pred[2] 
        l2p = y_pred[3]

        img1 = self.model.get_layer('input_1').output
        img2 = self.model.get_layer('input_2').output

        r1 = self.model.get_layer('reflectance_1').output
        l1 = self.model.get_layer('illumination_1').output
        r2 = self.model.get_layer('reflectance_2').output
        l2 = self.model.get_layer('illumination_2').output

        img1_pp = keras.layers.Multiply()([r1p, l2p])
        img2_pp = keras.layers.Multiply()([r2p, l1p])

        # reconstruction loss
        reconstruction_loss = K.mean(K.square(img1 - img1_pp)) + K.mean(K.square(img2 - img2_pp))

        # consistency loss
        consistency_loss = K.mean(K.square(l1 - l2p)) + K.mean(K.square(l2 - l1p)) + \
                            K.mean(K.square(r1 - r1p)) + K.mean(K.square(r2 - r2p))
    
        # entropy - convert to greyscale, then find bins?
        entropy_loss = self.getEntropy(r1) + self.getEntropy(r2)
        
        # sobel filter
        l1_sobelFilter = self.getSobelFilter(l1)
        l2_sobelFilter = self.getSobelFilter(l2)
        
        l1f = K.mean(K.depthwise_conv2d(l1, l1_sobelFilter))
        l2f = K.mean(K.depthwise_conv2d(l2, l2_sobelFilter))
        
        smoothness_loss = l1f + l2f
        
        weights = K.constant([0.7, 0.1, 0.1, 0.1], dtype=tf.float32)
        losses = K.variable([reconstruction_loss, consistency_loss, entropy_loss, smoothness_loss], dtype=tf.float32)
        total_loss = tf.tensordot(weights, losses, axes=1)
        return total_loss

    # *********** Helper functions ****************
    # returns an encoder stack
    def encoder(self, input_img, name, activation='selu'):
        
        # encoder layer
        x = Conv2D(16, (3, 3), activation=activation, padding='same')(input_img)
        x = Conv2D(16, (3, 3), activation=activation, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), activation=activation, padding='same')(x)
        x = Conv2D(32, (3, 3), activation=activation, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), activation=activation, padding='same')(x)
        x = Conv2D(64, (3, 3), activation=activation, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (1, 1), activation='sigmoid', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same', name=name)(x)
        
        return encoded

    # returns a decoder stack
    def decoder(self, encoded, name, activation='selu'):
        
        # decoder layer
        x = Conv2D(64, (1, 1), activation=activation, padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), activation=activation, padding='same')(x)
        x = Conv2D(64, (3, 3), activation=activation, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), activation=activation, padding='same')(x)
        x = Conv2D(32, (3, 3), activation=activation, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(16, (3, 3), activation=activation, padding='same')(x)
        x = Conv2D(16, (3, 3), activation=activation, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name=name)(x)
        
        return decoded

    # ***************** Helper Loss Functions ************************
    def log2(self, tensor):
        base = 2
        return tf.math.log(tensor) / tf.math.log(tf.constant(base, dtype=tf.float32))

    def toGrey(self, img):
        # returns a 2D grey tensor of the given 4D tensor
        img = K.squeeze(img, 0)
        img = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
        return img


    def getEntropy(self, inputTensor):
        
        inputTensor = K.squeeze(inputTensor, 0)

        # calculates probability in the r^3 cube
        n = max(K.int_shape(inputTensor))
        x = inputTensor[:, :, 0] + n * inputTensor[:, :, 1] + n**2 * inputTensor[:, :, 2]
        
        x = K.flatten(K.round(inputTensor * 255.))
        
        y, idx, count = tf.unique_with_counts(x)
        prob = tf.cast(count / tf.shape(count), tf.float32)
        
        # follows the entropy formula by multiplying counts by their respective probabilities
        entropy = -1 * tf.math.reduce_sum(tf.multiply(tf.math.multiply(self.log2(prob), prob), tf.cast(count, tf.float32)))
        return entropy

    # Maybe use 2 1D filter instead of a 2D filter?
    def getSobelFilter(self, inputTensor):
        
        sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                        [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                        [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])
        input_channels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
        sobelFilter = sobelFilter * input_channels
        
        return sobelFilter

if __name__ == "__main__":
    cg = CycleGAN()
    cg.model.summary()