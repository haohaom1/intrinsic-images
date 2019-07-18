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
from models.simpleJanknet.simple_janknet import SimpleJankNet

def main(path_imap, path_mmap, batch_size, num_epochs, model_name, num_imaps_per_mmap, hist_path=None):

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
    elif model_name == 'simpleJanknet':
        net = SimpleJankNet()
    else:
        print(f"model name {model_name} not found")
        exit(-1)

    print(f"model name is {model_name}")
    net.model.summary()

    num_list_imap = len([x for x in os.listdir(path_imap) if x.endswith('npy')])
    num_list_mmap = len([x for x in os.listdir(path_mmap) if x.endswith('npy')])

    LEN_DATA = min(num_list_imap, num_list_mmap * num_imaps_per_mmap)
    print("number of samples in data ", LEN_DATA)

    VALID_LEN_DATA = LEN_DATA - LEN_DATA % batch_size
        
    curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    # checkpoint
    filepath= f"weights-{model_name}" + "-{epoch:02d}-{loss:.2f}_" + curtime + ".hdf5"

    full_filepath = os.path.join(f"./models/{model_name}/", filepath)
    # save the minimum loss
    checkpoint = ModelCheckpoint(full_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # Fit the model
    history_obj = net.train(VALID_LEN_DATA, batch_size, num_epochs, data_gen.generator(path_imap, path_mmap, VALID_LEN_DATA, num_imaps_per_mmap=num_imaps_per_mmap), callbacks_list)
    # save the history object to a pickle file

    if not hist_path:
        hist_path = model_name
    json.dump(history_obj.history, open(os.path.join(f"./models/{model_name}", hist_path + "_" + curtime), "w"))
    final_epoch_fpath = os.path.join(f"./models/{model_name}", f"final_epoch_weights_{curtime}.hdf5")
    print(f"saving model to {final_epoch_fpath}")
    net.model.save(final_epoch_fpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path_imap', help='directory where the imap npy files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('path_mmap', help='directory where the imap files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('batch_size', help='calculate ambient and direct store imap', default=64, type=int)
    parser.add_argument('num_epochs', help='number of epochs to train - irrelevant if in test mode', default=20, type=int)
    parser.add_argument('num_imaps_per_mmap', help="number of imaps per mmap - irrelevant if in train mode", type=int, default=5)
    parser.add_argument('model_name', help="the name of the model")
    parser.add_argument('--hist_path', '-p', help='name of the history object, saved in the same path as this file')

    args = parser.parse_args()
    args = vars(args)

    main(**args)
