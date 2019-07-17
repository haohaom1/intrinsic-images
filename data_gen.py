import numpy as np 
import sys
import os
import random
import argparse
import cv2


# NOTE: zip takes care of the nondivisible issue, anything that is extra 
# from either list is truncated off 


def generator(imap_files, mmap_files, valid_len_data, log=False, resolution=128, batch_size=64):

    '''
    Takes two paths, and creates a generator to yield batches of data
    Can be used for training and validation data
    # Arguments
        imap_files = a list of filenames (strings) of npy files each containing an illumination map
        mmap_files = a list of filenames (strings) of npy files, each containing a material map
            Note that mmap_files has the number of imaps_per_mmap PRE-MULTIPLIED, so there will be NUM_IMAPS_PER_MMAP copies of each mmap npy
        valid_len_data = the valid length of the data, should be fully divisible by batch size
        log = default False. If true, then it takes the log of all the images, and adds a small shift to ensure the values are > 1
        resolution = default 128, square size of the image
        batch_size = default 64. batch size to use for training
    '''

    # maxlen will be the length of the zip
    zip_len = min(len(mmap_files), len(imap_files))

    rem = zip_len % batch_size
    max_len = zip_len - rem

    assert(valid_len_data == max_len)

    assert(max_len % batch_size == 0)

    number_of_batches = int(valid_len_data / batch_size)

    while True:

        # the start of the next epoch

        # shuffle the two lists
        random.shuffle(mmap_files)
        random.shuffle(imap_files)

        # this generates an iterable zip (iterables are generators)
        z = zip(all_mmap_files, imap_files)

        # this is for one epoch: always ensure that the number of samples in an epoch
        # is fully divisible by batch size
        # so each batch is the same size
        for i in range(number_of_batches):
            # this is for one batch
            batch_files = [next(z) for _ in range(batch_size)]
            # process the batch
            batch_res = []
            batch_imap = []
            batch_mmap = []
            for file_mmap, file_imap in batch_files:

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

                # resize by rescaling
                res_cropped = cv2.resize(res, (resolution, resolution), interpolation=cv2.INTER_AREA)
                imap_cropped = cv2.resize(imap, (resolution, resolution), interpolation=cv2.INTER_AREA)
                # mmap_cropped = cv2.resize(mmap, (resolution, resolution), interpolation=cv2.INTER_AREA)

                batch_res.append(res_cropped)
                batch_imap.append(imap_cropped)
                # batch_mmap.append(mmap_cropped)
            
            # in the future, need to return 5, need to take account of this inside model.train()
            yield np.array(batch_res), np.array(batch_imap) #, np.array(batch_mmap)


            