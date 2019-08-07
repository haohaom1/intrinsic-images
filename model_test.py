'''
Test file that displays the model predictions of imaps
'''

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
import cv2
import os
import matplotlib.gridspec as gridspec
import sys
import json

# import every type of model
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


def main(argv):
    '''
        angle brackets denote required arguments
        square brackets denote optional arguments
        as is the UNIX convention :)
    '''
    if len(argv) < 5:
        print('<model_name> <model_path> <path_imap> <path_mmap> [history_path]')
        return

    model_name = argv[1]

    if model_name == "janknet":
        net = JankNet()
    elif model_name == 'unet':
        net = UNet()
    elif model_name == 'simpleJanknet':
        net = SimpleJankNet()
    elif model_name == 'janknet2head':
        net = JankNet2Head()
    elif model_name == "strongerJanknet":
        net = StrongerJankNet()
    elif model_name == "brucenet":
        net = BruceNet()
    elif model_name == "dualunet":
        net = DualUNet()
    elif model_name == 'testJanknet':
        net = TestJankNet()
    else:
        print(f"model name {model_name} not found")
        exit(-1)


    model_path = argv[2]
    model = keras.models.load_model(model_path, custom_objects=net.custom_loss())
    # model = keras.models.load_model(model_path, custom_objects={'imap_loss': imap_loss, 'mmap_loss': mmap_loss})

    path_imap = argv[3]
    path_mmap = argv[4]

    imap_list = [x for x in os.listdir(path_imap) if x.endswith('.npy')]
    mmap_list = [x for x in os.listdir(path_mmap) if x.endswith('.npy')]

    # number of samples to show
    num_to_show = min(len(mmap_list), 5)

    mmaps = [np.load(os.path.join(path_mmap, x)) for x in np.random.choice(mmap_list, size=num_to_show, replace=False)]
    imaps = [np.load(os.path.join(path_imap, x)) for x in np.random.choice(imap_list, size=num_to_show, replace=False)]

    NUM_ITEMS = 5

    if len(argv) == 5:
        history = None
    else:
        history_path = argv[5]
        history = json.load(open(history_path, "r"))

    plt.figure()
    gs1 = gridspec.GridSpec(num_to_show+1, NUM_ITEMS)
    gs1.update(wspace=0.025, hspace=0.15) # set the spacing between axes. 

    for i, (mmap, imap) in enumerate(zip(imaps, mmaps)):
        axRow = [plt.subplot(gs1[i, j]) for j in range(NUM_ITEMS)]
        imap = cv2.resize(imap, (128, 128), interpolation=cv2.INTER_AREA)
        mmap = cv2.resize(mmap, (128, 128), interpolation=cv2.INTER_AREA)
        
        res = np.clip(imap * mmap, 0, 1)[np.newaxis, :]
        predImap, predMmap = model.predict(res)

        # pred mmap should be result divded by twice the predicted imap
        # predMmap = res / (predImap * 2)
        
        labels = ['mmap', 'imap', 'result', 'predImap', 'predMmap']

        # NOTE
        # imap is shown to be divided by 2 because it is between 0-2
        # predImap is half of the inputed imap
        # shows the clip result
        to_plot = [imap/2., mmap, res, predImap, predMmap]

        # apply gamma correction
        to_plot = [gamma_correction(x) for x in to_plot]
        
        for ax, l, p in zip(axRow, labels, to_plot):

            # only label if it's first row
            if i == 0:
                ax.set_title(l)
            ax.imshow(p.squeeze())
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if history:
        ax = plt.subplot(gs1[-1, :])
        ax.plot(history['loss'], label='loss')
        ax.plot(history['val_loss'], 'r', label='val_loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()

    # if it's Allen's computer, save the figure because matplotlib doesn't work because he doesn't have anaconda :(
    plt.savefig("model_test.png")

    plt.show()

# function names should match with the names of the corresponding output layers
def imap_loss(true_img, pred_img):

    imap_diff = K.mean(K.square((0.5 * true_img) - pred_img))
    return imap_diff

def mmap_loss(true_img, pred_img):

    mmap_diff = K.mean(K.square(true_img - pred_img))
    return mmap_diff

def gamma_correction(img, gamma=2.2):
    # assumes that image is a floating point from 0 to 1
    inv_gamma = 1.0 / gamma
    corrected = img ** inv_gamma
    return corrected

if __name__ == "__main__":
    main(sys.argv)