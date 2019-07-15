import numpy as np 
import sys
import os
import random
import argparse
import cv2

# NOTE: hardcoding is as follows
# imap_npy has 4800 images
# mmap_npy has 1200 images
# so by design, we multiply mmap * 4
# batch size 64 bc 4800 / 64 = 75 which is cleanly divisible
# file paths are hardcoded for the linux dwarves
# - Allen 2 July

# NOTE: zip takes care of the nondivisible issue, anything that is extra 
# from either list is truncated off 

def generator(path_imap, path_mmap, log=False, num_imaps_per_mmap=4, resolution=128):

    '''
    Takes two paths, and creates a generator
    '''
    # assert that the path exists
    assert os.path.isdir(path_imap) and os.path.isdir(path_mmap)

    imap_files = [x for x in os.listdir(path_imap) if x.endswith('npy')]
    mmap_files = [x for x in os.listdir(path_mmap) if x.endswith('npy')]

    while True:
        all_mmap_files = mmap_files * num_imaps_per_mmap    # default to 4 copies of list of mm

        # shuffle the two lists
        random.shuffle(all_mmap_files)
        random.shuffle(imap_files)
    
        # without replacement
        for file_mmap, file_imap in zip(all_mmap_files, imap_files):


            # ASSUMES THAT THE PATH STRUCTURE IS NAMING CONVENTION: NAMING CONVENTION: ./data/imap/[imap_npy, imap_npy_ambient, imap_npy_direct]/[train, test]/[gen_type]%d.npy
            # amb_imap = np.load(os.path.join(path_imap.replace('imap_npy', 'imap_npy_ambient'), file_imap), allow_pickle=True)
            # dir_imap = np.load(os.path.join(path_imap.replace('imap_npy', 'imap_npy_direct'), file_imap), allow_pickle=True)

            mmap = np.load(os.path.join(path_mmap, file_mmap), allow_pickle=True)
            imap = np.load(os.path.join(path_imap, file_imap), allow_pickle=True)
                
            assert(mmap.shape == imap.shape)
            res = np.multiply(mmap, imap)  # element wise multiplication

            # cutoff between 0 and 1
            # because image values can only be between 0 and 1
            # but real-world data can be larger
            res = np.clip(res, 0., 1.)

            # if using logspace, convert to 16 bit ints, add offset, then take log
            if log:
                offset = 5
                res = res * 65535 + offset

                res = np.log(res)

            # # if using linear space, scale all values between [-0.5, 0.5]
            # else:
            #     res -= 0.5

            assert(res.shape == imap.shape)

            # rescale
            imap_cropped = cv2.resize(imap, (resolution, resolution), interpolation=cv2.INTER_AREA)
            res_cropped = cv2.resize(res, (resolution, resolution), interpolation=cv2.INTER_AREA)

            # yield amb_imap, dir_imap, imap, mmap, res
            yield res, imap, mmap

def generator_batch(gen, batch_size=64):

    # gets list of res, imaps, mmaps with size = batch_size
    results = [next(gen) for _ in range(batch_size)]

    # splits up the lists for unpacking and turns them into np arrays
    final = tuple([np.array(l) for l in zip(*results)])

    return final