'''
    Allen Ma, Mike Fu
    Summer Research 2019
    Driver file to train the models
'''

import sys
import os
import json, datetime
import keras
from keras.callbacks import ModelCheckpoint
import data_gen
import argparse

# hardcoded
from models.janknet.janknet_separation import JankNet
from models.unet.unet_separation import UNet

def main(path_imap, path_mmap, batch_size, num_epochs, model_name, num_imaps_per_mmap, hist_path=None, save_all=False):

    if not os.path.isdir(path_imap):
        print(f"{path_imap} not a valid directory")
        exit(-1)
    if not os.path.isdir(path_mmap):
        print(f"{path_mmap} not a valid directory")
        exit(-1)

    if num_imaps_per_mmap <= 0:
        print(f"ratio: num imaps {num_imaps_per_mmap} must be greater than 0")
        exit(-1)

    if model_name == "janknet":
        net = JankNet()
    elif model_name == 'unet':
        net = UNet()
    else:
        print(f"model name {model_name} not found")
        exit(-1)

    num_list_imap = len([x for x in os.listdir(path_imap) if x.endswith('npy')])
    num_list_mmap = len([x for x in os.listdir(path_mmap) if x.endswith('npy')])

    LEN_DATA = min(num_list_imap, num_list_mmap * num_imaps_per_mmap)
    print("number of samples in data ", LEN_DATA)
        
    curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    # checkpoint
    filepath= f"./weights-{model_name}" + "-{epoch:02d}-{loss:.2f}_" + curtime + ".hdf5"
    # save the minimum loss
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=(not save_all))
    callbacks_list = [checkpoint]
    # Fit the model
    history_obj = net.train(LEN_DATA, batch_size, num_epochs, data_gen.generator(path_imap, path_mmap, num_imaps_per_mmap=num_imaps_per_mmap), callbacks_list)
    # save the history object to a pickle file

    if hist_path:
        json.dump(history_obj.history, open(hist_path + "_" + curtime, "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path_imap', help='directory where the imap npy files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('path_mmap', help='directory where the imap files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('batch_size', help='calculate ambient and direct store imap', default=64, type=int)
    parser.add_argument('num_epochs', help='number of epochs to train - irrelevant if in test mode', default=25, type=int)
    parser.add_argument('num_imaps_per_mmap', help="number of imaps per mmap - irrelevant if in train mode")
    parser.add_argument('model_name', help="the name of the model")
    parser.add_argument('--hist_path', '-p', help='name of the history object, saved in the same path as this file')
    parser.add_argument('--save_all', '-s', help="save weights of all epochs if this flag is set", action='store_true')

    args = parser.parse_args()
    args = vars(args)

    main(**args)