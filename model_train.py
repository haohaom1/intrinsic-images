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
import re


# hardcoded
from models.janknet.janknet_separation import JankNet
from models.unet.unet_separation import UNet
from models.simpleJanknet.simple_janknet import SimpleJankNet
from models.janknet2head.janknet2head import JankNet2Head
from models.mikenet.mikenet import MikeNet
from models.strongerJanknet.strongerjanknet import StrongerJankNet
from models.brucenet.brucenet import BruceNet
from models.testJanknet.testjank3 import TestJankNet
from models.dualunet.dualunet import DualUNet

from clr_callback import CyclicLR

# hardcoded training log file
TRAINING_LOG_PATH = "./models/training_log.csv"

def main(path_imap, path_mmap, batch_size, num_epochs, model_name, num_imaps_per_mmap, 
    hist_path=None, validation_split=0.2, no_validation=False, 
    inputs_to_network="result", ground_truth="imap,mmap", resolution=128, 
    gpu=0, load_weights=None, cyclic_lr=False, base_lr=1e-3, max_lr=6e-3):

    # change gpu id
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)   # should be 0 or 1

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
    net = None
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
    elif model_name == "dualunet":
        net = DualUNet(input_size=input_size)
    elif model_name == "testJanknet":
        net = TestJankNet(input_size=input_size)
    else:
        print(f"model name {model_name} not found")
        exit(-1)

    print(f"model name is {model_name}")
    net.model.summary()

    # saves the model architecture if doesn't exist already
    net.save_model_architecture(model_name, path=f'./models/{model_name}')

    if load_weights:
        net.load_weights(load_weights)
    
    curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # make a directory for this current instance
    new_dir = f'./models/{model_name}/instance_{curtime}'

    # add additional naming convention for retraining models
    if load_weights:
        old_instance = re.findall('instance.+?(?=/)', load_weights)     # grabs the instance_{curtime}
        new_dir = f'./models/{model_name}/retrained_{old_instance}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

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

    # number of batch updates
    batch_updates_per_epoch = VALID_LEN_DATA / batch_size

    ####### SETUP CALLBACKS


    # checkpoint
    filepath = f"weights-{model_name}" + "-{epoch:02d}-{loss:.2f}" + ".hdf5"

    full_filepath = os.path.join(new_dir, filepath)
    # this checkpoint only saves losses that have improved
    checkpoint = ModelCheckpoint(full_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # add a csv logger file that tracks metrics as training progresses
    csvlogger = CSVLogger(os.path.join(new_dir, f"rolling_log-{model_name}-{curtime}.csv"))

    # find a good learning rate if specified

    # whether or not to use cyclic learning rates
    # step size is the number of batch_updates per half cycle
    # Leslie Smith (author of cyclic policy paper suggests 2-8 * number of batch_updates), here we choose 4
    if cyclic_lr:
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size= 4 * batch_updates_per_epoch, mode='triangular2')
        callbacks_list = [checkpoint, csvlogger, clr]
    else:
        callbacks_list = [checkpoint, csvlogger]

    ###### CALL TRAIN

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
    parser.add_argument('--gpu', help='which gpu to use', type=int, default=0)
    parser.add_argument('--load_weights', '-lm', help='optionally load in model weights')
    parser.add_argument('--cyclic_lr', '-clr', help='uses cyclic learning rate policy to train network - note that it always uses the "decreasing triangular policy', action='store_true')
    parser.add_argument('--base_lr', '-blr', help='base LR for cyclic policy; if cyclic policy is False, then this parameter is irrelevant', type=float)
    parser.add_argument('--max_lr', '-mlr', help='max LR for cyclic policy; if cyclic policy is False, then this parameter is irrelevant', type=float)

    args = parser.parse_args()
    args = vars(args)

    main(**args)
