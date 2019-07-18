'''

Driver file to train models for the linux dwarves

'''

import os
import sys

def main(argv):
    if len(argv) < 6:
        print("[user] [device name] [batch size] [epochs] [num_imaps_per_mmap] [model name] [no_validation (true = no_validation)]")

    user = argv[1]
    device = argv[2]
    batch = argv[3]
    epoch = argv[4]
    num = argv[5]
    model = argv[6]
    no_validation = argv[7]


    if no_validation == "true":
        callstring = f"python3 model_train.py --no_validation /media/{user}/{device}/intrinsic-images/data/imap/imap_npy/train/ \
        /media/{user}/{device}/intrinsic-images/data/mmap/mmap_npy/train/ {batch} {epoch} {num} {model}"
    else:
        callstring = f"python3 model_train.py /media/{user}/{device}/intrinsic-images/data/imap/imap_npy/train/ \
        /media/{user}/{device}/intrinsic-images/data/mmap/mmap_npy/train/ {batch} {epoch} {num} {model}"
        
    os.system(callstring)

if __name__ == "__main__":
    main(sys.argv)