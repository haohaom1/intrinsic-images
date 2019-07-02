import numpy as np 
import sys
import os
import imageio
import random
import argparse

def generator(path_imap, path_mmap, log=False, num_imaps_per_mmap=5, resolution=128):

    '''
    Takes two paths, and creates a generator
    '''

    # assert that the path exists
    assert os.path.isdir(path_imap) and os.path.isdir(path_mmap)

    imap_files = [x for x in os.listdir(path_imap) if x.endswith('npy')]
    mmap_files = [x for x in os.listdir(path_mmap) if x.endswith('npy')]

    while True:
        all_mmap_files = mmap_files * num_imaps_per_mmap    # default to 5 copies of list of mm

        # shuffle the two lists
        random.shuffle(all_mmap_files)
        random.shuffle(imap_files)
    
        # without replacement
        for file_mmap, file_imap in zip(all_mmap_files, imap_files):

            # ASSUMES THAT THE PATH STRUCTURE IS NAMING CONVENTION: ./data/imap_npy/[(amb, dir, final)]/[gen_type]%d.npy
            amb_imap = np.load(os.path.join(path_imap.replace('final', 'ambient'), file_imap), allow_pickle=True)
            dir_imap = np.load(os.path.join(path_imap.replace('final', 'direct'), file_imap), allow_pickle=True)

            mmap = np.load(os.path.join(path_mmap, file_mmap), allow_pickle=True)
            imap = np.load(os.path.join(path_imap, file_imap), allow_pickle=True)
                
                
            res = np.multiply(mmap, imap)  # element wise multiplication

            # cutoff between 0 and 1
            res = np.clip(res, 0., 1.)

            # if using logspace, convert to 16 bit ints, add offset, then take log
            if log:
                offset = 5
                res = res * 65535 + offset

                res = np.log(res)

            # if using linear space, scale all values between [-0.5, 0.5]
            else:
                res -= 0.5

            # interpolation to the right dimension

            yield amb_imap, dir_imap, imap, mmap, res

def augmentData():
    '''
    Augments the data in a variety of ways
    '''
    pass


# def main(args):
#     gen = generator(**args)

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('path_imap', help='directory of all the illumination map numpy arrays')
#     parser.add_argument('path_mmap', help='directory of all the material map numpy arrays')
#     parser.add_argument('-l', '--log', help='use logarithm space', action='store_true')
#     parser.add_argument('-n', '--num_imaps_per_mmap', help='number of illumination maps per material map', default=5, type=int)
#     args, extras = parser.parse_known_args()

#     main(args)