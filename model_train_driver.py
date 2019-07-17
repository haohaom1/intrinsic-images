'''

Driver file to train models for the linux dwarves

'''

import os
import sys

def main(argv):
    if len(argv) < 6:
        print("[user] [device name] [batch size] [epochs] [num_imaps_per_mmap] [model name]")

    user = argv[0]
    device = argv[1]
    batch = argv[2]
    epoch = argv[3]
    num = argv[4]
    model = argv[5]

    os.system(f"python3 model_train.py /media/{user}/{device}/intrinsic-images/data/imap/imap_npy/train/ \
        /media/{user}/{device}/intrinsic-images/data/mmap/mmap_npy/train/ {batch} {epoch} {num} {model}")

if __name__ == "__main__":
    main(sys.argv)