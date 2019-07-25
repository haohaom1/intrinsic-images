import matplotlib.pyplot as plt 
import numpy as np
import os
import random
import sys

def main(argv):

    if len(argv) < 2:
        print("[name of directoy]")
        return

    directory = argv[1]

    files = [x for x in os.listdir(directory) if ( x.endswith('.npy') and not x.startswith("IMG") )]

    random.shuffle(files)

    # check if each npy array is of the correct size

    idx = 0
    imgs = []
    for fname in files:
        print('checking', fname)
        img = np.load(os.path.join(directory, fname))
        assert img.shape == (512, 512, 3)

        if idx < 100:
            imgs.append(img)
            idx += 1

    fig, axes = plt.subplots(10, 10, figsize=(10,10))
    for i, (im, ax, f) in enumerate(zip(imgs, axes.flatten(), files)):
        print(f, i)
        ax.set_title(f)
        ax.imshow(im)

    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)