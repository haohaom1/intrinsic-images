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

    files = [x for x in os.listdir(directory) if ( x.endswith('.npy') and not x.startswith("IMG")) ]
    random.shuffle(files)
    files = files[:100]

    imgs = [np.load(os.path.join(directory, x), allow_pickle=True) for x in files ]
    print(len(imgs))

    fig, axes = plt.subplots(10, 10, figsize=(10,10))
    for i, (im, ax, f) in enumerate(zip(imgs, axes.flatten(), files)):
        print(f, i)
        ax.set_title(f)
        ax.imshow(im)

    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)