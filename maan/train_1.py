'''
	* Maan Qraitem
	* Colby College
'''

import os
from tqdm import tqdm
import sys
import xml.etree.ElementTree as etree
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

# import random
# random.seed(1299)
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
import keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint

input_shape = (512, 512, 3)
IMG_SIZE = 512
batch_size = 50
num_classes = 2

directory = "/storage/mqrait20/linear_images/Data"

if (len(sys.argv) < 2):
	print("Not enought arugments")
	quit()

kind = sys.argv[1]
EPOCHS = int(sys.argv[2])
log_check = int(sys.argv[3])
number = sys.argv[4]



train_data = np.load(os.path.join(directory,"train" + "_" + kind + "_" + number + ".npy"))
np.random.shuffle(train_data)


if (log_check == 1 or log_check == 2):
	X_train = np.array([i[0].astype(float) for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
	Y_train = np.array([i[1] for i in train_data])

else:
	X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
	Y_train = np.array([i[1] for i in train_data])

if (log_check == 1):
	for i in tqdm(range(len(X_train))):
		X_train[i][X_train[i] == 0] = 2
		X_train[i][X_train[i] == 1] = 2
		X_train[i] = np.log(X_train[i])



# print(len(X_train))
# for i in range(30,40): 
# 	print(Y_train[i])
# 	cv2.imshow('image',X_train[i, :, :, :])
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# quit()

print(X_train[0])

droprate=0.35
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

# model.add(Conv2D(256, (2, 2), activation='relu'))
# model.add(BatchNormalization())
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(droprate))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(2, activation='softmax'))

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
optimizer = keras.optimizers.SGD(lr=0.0001)

model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])


filepath="../models/model" + "_" + kind + "_" + str(log_check) + "_" + number +".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train, Y_train,
		  epochs=EPOCHS,
		  batch_size=batch_size,
		  validation_split = 0.1,
		  callbacks=callbacks_list)


steps = len(history.history['loss'])

plt.plot(range(steps), history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('steps')
plt.legend(['train'], loc='upper left')

plt.savefig(os.path.join("../plots", "accu_plot"+ "_" + kind + "_" + number + ".jpg"))

plt.close("all")

# "Loss"
plt.plot(range(steps),history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('steps')
plt.legend(['train'], loc='upper right')

plt.savefig(os.path.join("../plots", "loss_plot" + "_" + kind + "_" + number + ".jpg"))
