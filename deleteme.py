import matplotlib.pyplot as plt 
import numpy as np
import os
import random

directory = '/media/yma21/milton/intrinsic-images/data/imap/imap_npy/train'
files = os.listdir(directory)[:100]
random.shuffle(files)

imgs = [np.load(os.path.join(directory, x), allow_pickle=True) for x in files if x.endswith('.npy')]

fig, axes = plt.subplots(10, 10, figsize=(10,10))
for i, (im, ax, f) in enumerate(zip(imgs, axes.flatten(), files)):
    print(f, i)
    ax.set_title(f)
    ax.imshow(im)

plt.show()