# import the necessary packages
from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys, os
import argparse
import random

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

import data_gen

# code from https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
class LearningRateFinder:
    def __init__(self, net, stopFactor=4, beta=0.98):
        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        self.net = net
        self.model = self.net.model
        self.stopFactor = stopFactor
        self.beta = beta
 
        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []
 
        # initialize our learning rate multiplier, average loss, best
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0


    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
             "DataFrameIterator", "Iterator", "Sequence"]
 
        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses


    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
 
        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)
 
        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss
 
        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return
 
        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth
 
        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, gen, startLR, endLR, epochs,
        steps_per_epoch, batchSize=64,
        verbose=1):
        # reset our class-specific variables
        self.reset()
  
        # if we're using a generator and the steps per epoch is not
        # supplied, raise an error
        if steps_per_epoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * steps_per_epoch
 
        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
 
        K.set_value(self.model.optimizer.lr, startLR)

        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
            self.on_batch_end(batch, logs))
 
        # check to see if we are using a data iterator
        self.model.fit_generator(
            gen,
            steps_per_epoch = steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[callback])

    def plot_loss(self, skipBegin=10, skipEnd=1, title="lr finder"):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]
 
        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
 
        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)

        plt.savefig("lr_find.png")

def main(path_imap, path_mmap, batch_size, 
    num_epochs, num_imaps_per_mmap, model_name,
    ground_truth="imap,mmap", inputs_to_network="result", resolution=128, gpu=0):

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

    lrf = LearningRateFinder(net)

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
    
    random.shuffle(imap_files)
    random.shuffle(mmap_files)

    train_len_data = LEN_DATA

    VALID_LEN_DATA = train_len_data - train_len_data % batch_size

    print("[model_train.py] number of samples of training data", VALID_LEN_DATA)

    # make the training data the length of valid_len_data
    imap_files_train = imap_files[:VALID_LEN_DATA]
    mmap_files_train = mmap_files[:VALID_LEN_DATA]

    assert(len(imap_files_train) == len(mmap_files_train))
    assert(len(imap_files_train) == VALID_LEN_DATA)

    gen = data_gen.generator(imap_files_train, mmap_files_train, path_mmap, path_imap, inputs_to_network, ground_truth, batch_size=batch_size, resolution=resolution)

    lrf.find(
        gen,
        1e-10, 1e1,
        num_epochs,
        VALID_LEN_DATA / batch_size,
        batchSize=batch_size)
 
    # plot the loss for the various learning rates and save the
    # resulting plot to disk
    lrf.plot_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_imap', help='directory where the imap npy files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('path_mmap', help='directory where the imap files are located. For train, you should specify the train folder. Likewise for test.')
    parser.add_argument('batch_size', help='calculate ambient and direct store imap', default=64, type=int)
    parser.add_argument('num_epochs', help='number of epochs to find lr for', default=5, type=int)
    parser.add_argument('num_imaps_per_mmap', help="number of imaps per mmap - irrelevant if in train mode", type=int, default=5)
    parser.add_argument('model_name', help="the name of the model")
    parser.add_argument('--inputs_to_network', '-i', help='if this argument is specified, pass in a string of image types [ambient, direct, imap, mmap, result] delimited by commas', type=str, default="result")
    parser.add_argument('--ground_truth', '-g', help='if this argument is specified, pass in a string of image types [ambient, direct, imap, mmap, result] delimited by commas', type=str, default="imap,mmap")
    parser.add_argument('--resolution', '-r', help='the size of the input image', type=int, default=128)
    parser.add_argument('--gpu', help='which gpu to use', type=int, default=0)

    args = parser.parse_args()
    args = vars(args)

    main(**args)
