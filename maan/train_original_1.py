
import os 
import random 
import numpy as np 
from tqdm import tqdm
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

import xml.etree.ElementTree as etree
import cv2
import sys

input_shape = (512, 512, 1)
EPOCHS = 20
batch_size = 25

images_dir = "../Dataset_Camera/Dataset_JPEG"
labels_dir = "../labels"


images = os.listdir(images_dir)
labels = os.listdir(labels_dir)

IMG_SIZE = 512

log_check = int(sys.argv[1])


data = [] 

for i, image in enumerate(tqdm(images)):

	image_name = image.split(".")[0]
	if (image_name + ".xml" in labels):
		root = etree.parse(os.path.join(labels_dir, image_name + ".xml"))
		objects = root.findall('object')
		name = objects[0][1].text
		if (name == "sweedish fish 1"):
		# if (name == "sweedish fish 1"):
			img = cv2.imread(os.path.join(images_dir, image), 0)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			data.append([np.array(img), [1, 0,0,0]])

		elif (name == "sweedish fish 2"):
		# if (name == "sweedish fish 1"):
			img = cv2.imread(os.path.join(images_dir, image), 0)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			data.append([np.array(img), [0,1, 0,0]])

		elif (name == "sweedish fish 3"):
		# if (name == "sweedish fish 1"):
			img = cv2.imread(os.path.join(images_dir, image), 0)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			data.append([np.array(img), [0,0, 1, 0]])


		else:
			img = cv2.imread(os.path.join(images_dir, image), 0)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			data.append([np.array(img), [0,0, 0, 1]])

	
	else:

		img = cv2.imread(os.path.join(images_dir, image), 0)
		img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		data.append([np.array(img), [0, 0,0, 1]])


test_ids = random.sample(range(1, len(images)), int(0.1*len(images)))

train_data = [] 
test_data = [] 

for i in range(len(images)):

	if i in test_ids:
		test_data.append(data[i])
	else: 
		train_data.append(data[i])



if (log_check == 1):
	X_train = np.array([i[0].astype(float) for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	Y_train = np.array([i[1] for i in train_data])

	X_test = np.array([i[0].astype(float) for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	Y_test = np.array([i[1] for i in test_data])

else:
	X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	Y_train = np.array([i[1] for i in train_data])

	X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	Y_test = np.array([i[1] for i in test_data])



if (log_check == 1):
	for i in tqdm(range(len(X_train))):
		X_train[i][X_train[i] == 0] = 2
		X_train[i][X_train[i] == 1] = 2
		X_train[i] = np.log(X_train[i])

	for i in tqdm(range(len(X_test))):
		X_test[i][X_test[i] == 0] = 2
		X_test[i][X_test[i] == 1] = 2
		X_test[i] = np.log(X_test[i])


for i in range(4): 
    cv2.imshow('image',X_train[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(X_train[0])

droprate=0.25

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

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())


tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
optimizer = keras.optimizers.SGD(lr=0.0001)

model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

history = model.fit(X_train, Y_train,
		  epochs=EPOCHS,
		  batch_size=batch_size,
		  validation_split = 0.01,
		  callbacks=[tbCallBack])

print(model.evaluate(X_test, Y_test))

model.save("../models/model_original.h5")

