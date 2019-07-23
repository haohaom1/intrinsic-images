import numpy as np 
import sys
import os
import random
import argparse
import cv2


# NOTE: zip takes care of the nondivisible issue, anything that is extra 
# from either list is truncated off 

# Preprocesses an image to be ready to fit into the network
# takes in images (.npy) that are 512 * 512
# currently only
def preprocess(ambient, direct, imap, mmap, log, resolution):
    assert(mmap.shape == imap.shape)

    # resize by rescaling
    imap = cv2.resize(imap, (resolution, resolution), interpolation=cv2.INTER_AREA)
    mmap = cv2.resize(mmap, (resolution, resolution), interpolation=cv2.INTER_AREA)
    
    if ambient is not None:
        ambient = cv2.resize(ambient, (resolution, resolution), interpolation=cv2.INTER_AREA)
    if direct is not None:
        direct = cv2.resize(direct, (resolution, resolution), interpolation=cv2.INTER_AREA)

    res = np.multiply(mmap, imap)  # element wise multiplication

    # cutoff between 0 and 1
    # because image values can only be between 0 and 1
    # result must be between 0 and 1 
    # but imaps can be between 0 and 2
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

    return ambient, direct, imap, mmap, res



def generator(imap_files, mmap_files, path_mmap, path_imap,  
              inputs_to_network=None, ground_truth=None,
              log=False, resolution=128, batch_size=64):

    '''
    Takes two paths, and creates a generator to yield batches of data
    Can be used for training and validation data
    # Arguments
        imap_files = a list of filenames (strings) of npy files each containing an illumination map (pass in the actual imap, not ambient or direct)
        mmap_files = a list of filenames (strings) of npy files, each containing a material map
            NOTE that mmap_files has the number of imaps_per_mmap PRE-MULTIPLIED, so there will be NUM_IMAPS_PER_MMAP copies of each mmap npy
        inputs_to_network = a list of which types of images to yield in the generator that are fed into the network. 
                Select from ['ambient', 'direct', 'imap', 'mmap', 'result'].
                Default to returning the result
        ground_truth = a list of which types of images to yield in the generator that are used as the ground truth
                Select from ['ambient', 'direct', 'imap', 'mmap', 'result'].
                Default to returning the imap

        log = default False. If true, then it takes the log of all the images, and adds a small shift to ensure the values are > 1
        resolution = default 128, square size of the image
        batch_size = default 64. batch size to use for training
    '''

    valid_len_data = len(imap_files)

    while True:

        # the start of the next epoch

        # shuffle the two lists
        random.shuffle(mmap_files)
        random.shuffle(imap_files)

        # this generates an iterable zip (iterables are generators)
        z = zip(mmap_files, imap_files)

        number_of_batches = int(valid_len_data / batch_size)
        assert(number_of_batches == valid_len_data // batch_size)

        path_amb = path_imap.replace('imap_npy', 'imap_npy_ambient')
        path_dir = path_imap.replace('imap_npy', 'imap_npy_direct')

        # whether or not to use the imap components
        explicit_inputs_to_network = ['result'] if not inputs_to_network else inputs_to_network
        explicit_ground_truth = ['imap'] if not ground_truth else ground_truth
        use_ambient = 'ambient' in explicit_inputs_to_network or 'ambient' in explicit_ground_truth
        use_direct = 'direct' in explicit_inputs_to_network or 'direct' in explicit_ground_truth

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
            batch_amb = []
            batch_dir = []

            for file_mmap, file_imap in batch_files:

                mmap = np.load(os.path.join(path_mmap, file_mmap), allow_pickle=True)
                imap = np.load(os.path.join(path_imap, file_imap), allow_pickle=True)

                if use_ambient:
                    ambient = np.load(os.path.join(path_amb, file_imap), allow_pickle=True)
                else:
                    ambient = None

                if use_direct:
                    direct = np.load(os.path.join(path_dir, file_imap), allow_pickle=True)
                else:
                    direct = None
                

                # assert(mmap.shape == imap.shape)
                # res = np.multiply(mmap, imap)  # element wise multiplication

                # # cutoff between 0 and 1
                # # because image values can only be between 0 and 1
                # # result must be between 0 and 1 
                # # but imaps can be between 0 and 2
                # res = np.clip(res, 0., 1.)

                # # if using logspace, convert to 16 bit ints, add offset, then take log
                # if log:
                #     offset = 5
                #     res = res * 65535 + offset

                #     res = np.log(res)

                # # # if using linear space, scale all values between [-0.5, 0.5]
                # # else:
                # #     res -= 0.5

                # assert(res.shape == imap.shape)

                # # resize by rescaling
                # res_cropped = cv2.resize(res, (resolution, resolution), interpolation=cv2.INTER_AREA)
                # imap_cropped = cv2.resize(imap, (resolution, resolution), interpolation=cv2.INTER_AREA)
                # mmap_cropped = cv2.resize(mmap, (resolution, resolution), interpolation=cv2.INTER_AREA)
                # ambient_cropped = cv2.resize(ambient, (resolution, resolution), interpolation=cv2.INTER_AREA)
                # direct_cropped = cv2.resize(direct, (resolution, resolution), interpolation=cv2.INTER_AREA)

                ambient, direct, imap, mmap, res = preprocess(ambient, direct, imap, mmap, log, resolution)

                batch_res.append(res)
                batch_imap.append(imap)
                batch_mmap.append(mmap)

                if use_ambient:
                    batch_amb.append(ambient)
                if use_direct:
                    batch_dir.append(direct)

            # convert to npy arrays
            batch_res = np.array(batch_res)
            batch_imap = np.array(batch_imap)
            batch_mmap = np.array(batch_mmap)

            if use_ambient:
                batch_amb = np.array(batch_amb)
            if use_direct:
                batch_dir = np.array(batch_dir)

            inputs = []
            gtruth = []

            param_dict = dict(zip(['ambient', 'direct', 'imap', 'mmap', 'result'], 
                                  [batch_amb, batch_dir, batch_imap, batch_mmap, batch_res]))

            # loop through inputs_to_network to yield the correct inputs to network
            if not inputs_to_network:       # default value, inputs_to_network is None
                inputs = batch_res
            elif len(inputs_to_network) == 1:  # if there is only one value needed, dont need list
                inputs = param_dict[inputs_to_network[0]]
            else:
                for val in inputs_to_network:
                    inputs.append(param_dict[val])
            
            # repeat for ground truth
            if not ground_truth:       # default value, inputs_to_network is None
                gtruth = batch_imap
            elif len(ground_truth) == 1:  # if there is only one value needed, dont need list
                gtruth = param_dict[ground_truth[0]]
            else:
                for val in ground_truth:
                    gtruth.append(param_dict[val])
            
            yield inputs, gtruth

            # # in the future, need to return 5, need to take account of this inside model.train()
            # yield np.array(batch_res), np.array(batch_imap) #, np.array(batch_mmap)


            