'''
Test file that displays the model predictions of imaps
'''

import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
import cv2
import os
import matplotlib.gridspec as gridspec
import sys
import json

def imap_only_loss(true_img, pred_img):
    return K.mean(K.square(0.5 * true_img - pred_img))

def main(argv):
    '''
        angle brackets denote required arguments
        square brackets denote optional arguments
        as is the UNIX convention :)
    '''
    if len(argv) < 4:
        print('<model_path> <path_imap> <path_mmap> [history_path]')
        return

    model_path = argv[1]
    model = keras.models.load_model(model_path, custom_objects={'imap_only_loss': imap_only_loss})

    path_imap = argv[2]
    path_mmap = argv[3]

    imap_list = [x for x in os.listdir(path_imap) if x.endswith('.npy')]
    mmap_list = [x for x in os.listdir(path_mmap) if x.endswith('.npy')]

    # number of samples to show
    num_to_show = min(len(mmap_list), 5)

    mmaps = [np.load(os.path.join(path_mmap, x)) for x in np.random.choice(mmap_list, size=num_to_show, replace=False)]
    imaps = [np.load(os.path.join(path_imap, x)) for x in np.random.choice(imap_list, size=num_to_show, replace=False)]

    NUM_ITEMS = 5

    if len(argv) == 4:
        history = None
    else:
        history_path = argv[4]
        history = json.load(open(history_path, "r"))

    plt.figure()
    gs1 = gridspec.GridSpec(num_to_show+1, NUM_ITEMS)
    gs1.update(wspace=0.025, hspace=0.15) # set the spacing between axes. 

    for i, (mmap, imap) in enumerate(zip(imaps, mmaps)):
        axRow = [plt.subplot(gs1[i, j]) for j in range(NUM_ITEMS)]
        imap = cv2.resize(imap, (128, 128), interpolation=cv2.INTER_AREA)
        mmap = cv2.resize(mmap, (128, 128), interpolation=cv2.INTER_AREA)
        
        res = np.clip(imap * mmap, 0, 1)[np.newaxis, :]
        predImap = model.predict(res)

        # pred mmap should be result divded by twice the predicted imap
        predMmap = res / (predImap * 2)
        
        labels = ['mmap', 'imap', 'result', 'predImap', 'predMmap']

        # NOTE
        # imap is shown to be divided by 2 because it is between 0-2
        # predImap is half of the inputed imap
        to_plot = [imap/2., mmap, res, predImap, predMmap]
        
        for ax, l, p in zip(axRow, labels, to_plot): 
            ax.imshow(p.squeeze())
            ax.set_title(l)
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

    plt.show()

if __name__ == "__main__":
    main(sys.argv)