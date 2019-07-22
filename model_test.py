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

def imap_only_loss(true_img, pred_img):
    return K.mean(K.square(0.5 * true_img - pred_img))


def main(argv):

    if len(argv) < 5:
        print('[model_path] [path_imap] [path_mmap] [history_path]')
        return

    model_path = argv[1]
    model = keras.models.load_model(model_path, custom_objects={'imap_only_loss': imap_only_loss})

    path_imap = argv[2]
    path_mmap = argv[3]

    imap_list = [x for x in os.listdir(path_imap) if x.endswith('.npy')]
    mmap_list = [x for x in os.listdir(path_mmap) if x.endswith('.npy')]

    num_to_show = min(len(mmap_list), 3)

    mmaps = [np.load(os.path.join(path_mmap, x)) for x in mmap_list]
    imaps = [np.load(os.path.join(path_imap, x)) for x in np.random.choice(imap_list, size=num_to_show, replace=False)]

    NUM_ITEMS = 5


    histoy_path = argv[4]
    history = json.load(open(histoy_path, "r"))

    plt.figure()
    gs1 = gridspec.GridSpec(num_to_show+1, 5)
    gs1.update(wspace=0.025, hspace=0.15) # set the spacing between axes. 

    for i, (mmap, imap) in enumerate(zip(imaps, mmaps)):
        axRow = [plt.subplot(gs1[i, j]) for j in range(5)]
        imap = cv2.resize(imap, (128, 128), interpolation=cv2.INTER_AREA)
        mmap = cv2.resize(mmap, (128, 128), interpolation=cv2.INTER_AREA)
        
        res = np.clip(imap * mmap, 0, 1)[np.newaxis, :]
        predImap = model.predict(res) * 2
        predMmap = res / predImap
        
        labels = ['mmap', 'imap', 'result', 'predImap', 'predMmap']
        to_plot = [imap, mmap, res, predImap, predMmap]
        
        for ax, l, p in zip(axRow, labels, to_plot): 
            ax.imshow(p.squeeze())
            ax.set_title(l)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        
    ax = plt.subplot(gs1[-1, :])
    ax.plot(history['loss'], label='loss')
    ax.plot(history['val_loss'], 'r', label='val_loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    plt.show()

if __name__ == "__main__":
    main(sys.argv)