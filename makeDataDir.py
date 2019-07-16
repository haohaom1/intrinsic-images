'''
Author: Mike Fu

this file is used to create the skeleton of the data directory

passed in directory
-> imap
    -> imap_npy_ambient
        -> train
        -> test
    -> imap_npy_direct
        -> train
        -> test
    -> imap_npy
        -> train
        -> test
-> mmap
    -> mmap_npy
        -> train
        -> test

'''

import os
import sys

def main(argv):
    if len(argv) < 2:
        print('pass in root directory')
        return

    root_dir = argv[1]

    # if root doesn't exist, make it
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    types = ['imap', 'mmap']
    imaps = ['imap_npy_ambient', 'imap_npy_direct', 'imap_npy']
    tt = ['train', 'test']


    for typ in types:

        level0 = os.path.join(root_dir, typ)
        if not os.path.isdir(level0):
            os.mkdir(level0)

        # make imap directory
        if typ == 'imap':
            for im in imaps:
                level1 = os.path.join(level0, im)
                if not os.path.isdir(level1):
                    os.mkdir(level1)

                for t in tt:
                    level2 = os.path.join(level1, t)
                    if not os.path.isdir(level2):
                        os.mkdir(level2)

        # make mmap directory
        elif typ == 'mmap':

            level1 = os.path.join(level0, 'mmap_npy')
            if not os.path.isdir(level1):
                os.mkdir(level1)

            for t in tt:
                level2 = os.path.join(level1, t)
                if not os.path.isdir(level2):
                    os.mkdir(level2)

if __name__ == "__main__":
    main(sys.argv)