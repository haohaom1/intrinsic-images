'''
    Allen Ma, Mike Fu
    Summer Research 2019
    Driver file to train the models
'''

import sys
import os
import json, datetime
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
import data_gen
import argparse
import random

# dependency for writing csvs
# may consider just using csv package
import csv

# hardcoded
from models.janknet.janknet_separation import JankNet
from models.unet.unet_separation import UNet
from models.simpleJanknet.simple_janknet import SimpleJankNet
from models.janknet2head.janknet2head import JankNet2Head
from models.mikenet.mikenet import MikeNet
from models.strongerJanknet.strongerjanknet import StrongerJankNet
from models.brucenet.brucenet import BruceNet

# hardcoded training log file
TRAINING_LOG_PATH = "./models/training_log.csv"

def main(path_imap, path_mmap, batch_size, num_epochs, model_name, num_imaps_per_mmap, hist_path=None, validation_split=0.2, no_validation=False, inputs_to_network="", ground_truth="", resolution=128):

    if not os.path.isdir(path_imap):
        print(f"{path_imap} not a valid directory")
        exit(-1)
    if not os.path.isdir(path_mmap):
        print(f"{path_mmap} not a valid directory")
        exit(-1)

    if num_imaps_per_mmap <= 0:
        print(f"ratio: num imaps {num_imaps_per_mmap} must be greater than 0")
        exit(-1)

    input_size = (resolution, resolution, 3)
    print(f'input size: {input_size}')
    # determines model name
    if model_name == "janknet":
        net = JankNet(input_size=input_size)
    elif model_name == 'unet':
        net = UNet(input_size=input_size)
    elif model_name == 'simpleJanknet':
        net = SimpleJankNet(input_size=input_size)
    elif model_name == 'janknet2head':
        net = JankNet2Head(input_size=input_size)
    elif model_name == 'mikenet':
        net = MikeNet(input_size=input_size)
    elif model_name == "strongerJanknet":
        net = StrongerJankNet(input_size=input_size)
    elif model_name == "brucenet":
        net = BruceNet(input_size=input_size)
    else:
        print(f"model name {model_name} not found")
        exit(-1)

    print(f"model name is {model_name}")
    net.model.summary()
    
    curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # make a directory for this current instance
    new_dir = f'./models/{model_name}/instance_{curtime}'
    os.makedirs(new_dir)

    # checkpoint
    filepath = f"weights-{model_name}" + "-{epoch:02d}-{loss:.2f}" + ".hdf5"

    full_filepath = os.path.join(new_dir, filepath)
    # this checkpoint only saves losses that have improved
    checkpoint = ModelCheckpoint(full_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # add a csv logger file that tracks metrics as training progresses
    csvlogger = CSVLogger(os.path.join(new_dir, f"rolling_log-{model_name}-{curtime}.csv"))

    callbacks_list = [checkpoint, csvlogger]

    # pass in the names of files beforehand
    # assert that the path exists
    assert os.path.isdir(path_imap) and os.path.isdir(path_mmap)

    imap_files = [x for x in os.listdir(path_imap) if x.endswith('npy')]
    mmap_files = [x for x in os.listdir(path_mmap) if x.endswith('npy')]

    mmap_files = mmap_files * num_imaps_per_mmap 
    LEN_DATA = min(len(imap_files), len(mmap_files))

    # check that each element in input images is valid
    inputs_to_network = inputs_to_network.split(",")
    ground_truth = ground_truth.split(",")

    print("input types are", inputs_to_network)
    print("ground truth types are", ground_truth)

    valid_images = ['ambient', 'direct', 'imap', 'mmap', 'result']
    for i in inputs_to_network:
        if i not in valid_images:
            raise Exception(f"{i} is not a valid type for input to network")
    for i in ground_truth:
        if i not in valid_images:
            raise Exception(f"{i} is not a valid type for ground truth")
    

    if no_validation:
        validation_split = 0

    validation_len_data = int(validation_split * LEN_DATA)
    train_len_data = LEN_DATA - validation_len_data

    random.shuffle(imap_files)
    random.shuffle(mmap_files)

    imap_files_train = imap_files[validation_len_data:]
    imap_files_validation = imap_files[:validation_len_data]

    mmap_files_train = mmap_files[validation_len_data:]
    mmap_files_validation = mmap_files[:validation_len_data]

    VALID_LEN_DATA = train_len_data - train_len_data % batch_size
    VALID_VALIDATION_LEN_DATA = validation_len_data - validation_len_data % batch_size

    if no_validation:
        print("not using validation")
    else:
        print("using validation")
    print("[model_train.py] number of samples of training data", VALID_LEN_DATA)
    print("[model_train.py] number of samples of validation data", VALID_VALIDATION_LEN_DATA)

    # make the validation data the length of valid_validation_len_data
    imap_files_validation = imap_files_validation[:VALID_VALIDATION_LEN_DATA]
    mmap_files_validation = mmap_files_validation[:VALID_VALIDATION_LEN_DATA]

    # make the training data the length of valid_len_data
    imap_files_train = imap_files_train[:VALID_LEN_DATA]
    mmap_files_train = mmap_files_train[:VALID_LEN_DATA]

    assert(len(imap_files_train) == len(mmap_files_train))
    assert(len(imap_files_validation) == len(mmap_files_validation))

    assert(len(imap_files_validation) == VALID_VALIDATION_LEN_DATA)
    assert(len(imap_files_train) == VALID_LEN_DATA)
    
    if no_validation:
        history_obj = net.train(VALID_LEN_DATA, batch_size, num_epochs, 
            data_gen.generator(imap_files_train, mmap_files_train, path_mmap, path_imap, inputs_to_network, ground_truth, batch_size=batch_size, resolution=resolution),
            validation_gen = None,
            validation_len_data = None,
            callbacks=callbacks_list)
    else:
        # Fit the model
        history_obj = net.train(VALID_LEN_DATA, batch_size, num_epochs, 
            data_gen.generator(imap_files_train, mmap_files_train, path_mmap, path_imap, inputs_to_network, ground_truth, batch_size=batch_size, resolution=resolution),
            validation_gen = data_gen.generator(imap_files_validation, mmap_files_validation, path_mmap, path_imap, inputs_to_network, ground_truth, batch_size=batch_size, resolution=resolution),
            validation_len_data = VALID_VALIDATION_LEN_DATA,
            callbacks=callbacks_list)
        # save the history object to a pickle file

    if not hist_path:
        hist_path = model_name
    json.dump(history_obj.history, open(os.path.join(new_dir, hist_path + "_" + curtime), "w"))
    final_epoch_fpath = os.path.join(new_dir, f"final_epoch_weights_{curtime}.hdf5")
    print(f"saving model to {final_epoch_fpath}")
    net.model.save(final_epoch_fpath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path_imap', help='directory where the imap npy files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('path_mmap', help='directory where the imap files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('batch_size', help='calculate ambient and direct store imap', default=64, type=int)
    parser.add_argument('num_epochs', help='number of epochs to train', default=20, type=int)
    parser.add_argument('num_imaps_per_mmap', help="number of imaps per mmap - irrelevant if in train mode", type=int, default=5)
    parser.add_argument('model_name', help="the name of the model")
    parser.add_argument('--hist_path', '-p', help='name of the history object, saved in the same path as this file')
    parser.add_argument('--validation_split', '-s', help='ratio of train/validation split 0.2 means 20 percent of data is set aside as validation data. is used as validation: default is use validation and 0.2', type=float, default=0.2)
    parser.add_argument('--no_validation', '-nv', help='if this flag is set, then there is NO validation set. The validation_split flag is disregarded in this case', action="store_true")
    parser.add_argument('--inputs_to_network', '-i', help='if this argument is specified, pass in a string of image types [ambient, direct, imap, mmap, result] delimited by commas', type=str)
    parser.add_argument('--ground_truth', '-g', help='if this argument is specified, pass in a string of image types [ambient, direct, imap, mmap, result] delimited by commas', type=str)
    parser.add_argument('--resolution', '-r', help='the size of the input image', type=int, default=128)

    args = parser.parse_args()
    args = vars(args)

    main(**args)
