import matplotlib.pyplot as plt 
import numpy as np
import os
import random

directory = '/media/milton/intrinsic-images/data/imap/imap_npy/train'
files = random.shuffle(os.listdir(directory))[:100]

imgs = [np.load(os.path.join(directory, x), allow_pickle=True) for x in files if x.endswith('.npy')]

fig, axes = plt.subplots(10, 10, figsize=(8,8))
for im, ax in zip(imgs, axes.flatten()):
    ax.imshow(im)