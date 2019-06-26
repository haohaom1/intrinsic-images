'''
	* Maan Qraitem
	* Colby College
'''

import os
import xml.etree.ElementTree as etree
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
import keras
from tqdm import tqdm
import sys

from keras.models import Sequential
from keras.models import load_model

input_shape = (512, 512, 3)
IMG_SIZE = 512
num_classes = 2
batch_size = 25

directory = "/storage/mqrait20/linear_images/Data"

if (len(sys.argv) < 2):
	print("Not enought arugments")
	quit()

kind = sys.argv[1]
log_check = int(sys.argv[2])
number = sys.argv[3]


test_data = np.load(os.path.join(directory, "test"+ "_" + kind + "_" + number + ".npy"))


if (log_check == 1 or log_check == 2):

	X_test = np.array([i[0].astype(float) for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
	Y_test = np.array([i[1] for i in test_data])

else:

	X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
	Y_test = np.array([i[1] for i in test_data])

info = np.array([i[2] + "_" + i[3] for i in test_data])


if (log_check == 1):
	print("Mamama")
	for i in tqdm(range(len(X_test))):
		X_test[i][X_test[i] == 0] = 2
		X_test[i][X_test[i] == 1] = 2
		X_test[i] = np.log(X_test[i])

# count = 0
# for i in range(len(X_test)):
# 	if test_data[i][3] != "normal": 
# 		print(test_data[i][3]) 
# 	if test_data[i][3] == "Red":
# 		print(Y_test[i])
# 		cv2.imshow('image',X_test[i, :, :, :])
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()
# 		count += 1 
# 		if count == 4: 
# 			quit() 

# quit()


droprate=0.25
model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(droprate))

# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(droprate))

# model.add(Conv2D(256, (2, 2), activation='relu'))
# model.add(BatchNormalization())
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(droprate))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(4, activation='softmax'))


tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
optimizer = keras.optimizers.SGD(lr=0.0001)

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])


model.load_weights(os.path.join("../models", "model" + "_" + kind + "_" + str(log_check) + "_" + number + ".h5"))

Accuracy_info = {}


print("-------- Genereating predictions --------")
#Calculates the predictions of each test file.
count_positives = 0
count = 0
for i, ele in enumerate(tqdm(X_test)):

	# cv2.imshow('image',ele)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# if i == 2:
	# 	quit() 

	data = ele.reshape(1, IMG_SIZE, IMG_SIZE, 3)

	prediction = model.predict([data])[0]

	if info[i] not in Accuracy_info:
		Accuracy_info[info[i]] = [0, 0]

	exp = info[i].split("_")[1]

	if np.argmax(prediction) == 0:
		count_positives+=1

	if (np.argmax(prediction) == np.argmax(Y_test[i])):
		count += 1
		Accuracy_info[info[i]][0] += 1

	else:
		Accuracy_info[info[i]][1] += 1


print("Accuracy for kind " + kind +  ": ", count/len(X_test))
# print(Accuracy_info)
# print("Percentage of positives: ", count_positives/len(X_test))



N = len(Accuracy_info)
trues = [Accuracy_info[i][0] for i in Accuracy_info]
falses = [Accuracy_info[i][1] for i in Accuracy_info]
ind = np.arange(N)    # the x locations for the groups
width = 0.8       # the width of the bars: can also be len(x) sequence

plt.figure(figsize=(75,12))

p1 = plt.bar(ind, trues, width)
p2 = plt.bar(ind, falses, width,
			 bottom=trues)

plt.ylabel("Count")
plt.title('Counts by exp/WB')
plt.xticks(ind, [i for i in Accuracy_info])
plt.yticks(np.arange(0, 100, 2))
plt.legend((p1[0], p2[0]), ('True', 'False'))

plt.savefig(os.path.join("../accuracy_plots", "acc_plot" + "_" + kind + "_" + str(log_check) + "_" + number + ".jpg"))

with open("../accuracy_text/accuracy_values_" + kind + ".txt", "a") as myfile:
	myfile.write(number + ":" + str(count/len(X_test)) + "\n")