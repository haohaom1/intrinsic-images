'''
    Allen Ma, Mike Fu
    Summer Research 2019
    Driver file to test the models
'''

import sys
import os
import json, datetime
import keras
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import data_gen
import argparse

# hardcoded
from models.janknet.janknet_separation import JankNet
from models.unet.unet_separation import UNet

def main(path_imap, path_mmap, batch_size, model_weights, num_imaps_per_mmap):

    if not os.path.isdir(path_imap):
        print(f"{path_imap} not a valid directory")
        exit(-1)
    if not os.path.isdir(path_mmap):
        print(f"{path_mmap} not a valid directory")
        exit(-1)

    if num_imaps_per_mmap <= 0:
        print(f"ratio: num imaps {num_imaps_per_mmap} must be greater than 0")
        exit(-1)

    model = load_model(model_weights, custom_objects={'imap_only_loss': imap_only_loss})

    LEN_DATA = min(len(os.listdir(path_imap)), len(os.listdir(path_mmap)) * num_imaps_per_mmap)
    print("number of samples in data ", LEN_DATA)
        
    output = model.evaluate(LEN_DATA, batch_size, data_gen(path_imap, path_mmap, num_imaps_per_mmap=NUM_MMAPS_PER_IMAP))

    print(output)

# hardcoded, find a way to integrate this somehow
def imap_only_loss(true_img, pred_img):
    return K.mean(K.square(true_img * 0.5 - pred_img))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path_imap', help='directory where the imap npy files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('path_mmap', help='directory where the imap files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('batch_size', help='calculate ambient and direct store imap', default=64, type=int)
    parser.add_argument('num_imaps_per_mmap', help="number of imaps per mmap - irrelevant if in train mode")
    parser.add_argument('model_weights', help="path to the weights file to reconstruct the model")

    args = parser.parse_args()
    args = vars(args)

    main(**args)