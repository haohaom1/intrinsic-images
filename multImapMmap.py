'''
Multiplies illumination map and material maps together. Assume Maps are Represented as numpy arrays (.npy)
'''

import numpy as np 
import sys
import os
import imageio

def main(argv):
    if len(argv) < 4:
        print('[path to illumination map] [path to material map] [path of product] [path of product as png (opt.)]')
        return

    imaps_per_mmap = 5

    path_imap = argv[1]
    path_mmap = argv[2]
    path_res = argv[3]

    # assert that the path exists
    assert os.path.isdir(path_imap) and os.path.isdir(path_mmap)

    if not os.path.isdir(path_res):
        os.mkdir(path_res)

    # optional folder for png's
    if len(argv) == 5:
        path_res_png = argv[4] 
        if not os.path.isdir(path_res_png):
            os.mkdir(path_res_png)


    imap_files = [x for x in os.listdir(path_imap) if x.endswith('npy')]
    range_imap_files = range(len(imap_files))
    
    # without replacement
    for file_mmap in os.listdir(path_mmap):
        if file_mmap.endswith('npy'):
            mmap = np.load(os.path.join(path_mmap, file_mmap), allow_pickle=True)

            # randomly pick from imaps (without replacement?)
            idxs = np.random.choice(range_imap_files, size=imaps_per_mmap, replace=False)
        
            # naming convention: mult-[nameOfImap]-[nameOfMmap]-d.npy
            for i, idx in enumerate(idxs):
                
                curr_imap = np.load(os.path.join(path_imap, imap_files[idx]), allow_pickle=True)
                res = np.multiply(mmap, curr_imap)  # element wise multiplication

                # cutoff between 0 and 1
                res = np.clip(res, 0., 1.)

                name_imap = imap_files[idx].split('.')[0]
                name_mmap = file_mmap.split('.')[0]

                fname = 'mult-{}-{}-{}'.format(name_imap, name_mmap, i)

                np.save(os.path.join(path_res, fname + '.npy'), res) # save numpy array
                imageio.imwrite(os.path.join(path_res_png, fname + '.png'), res)

if __name__ == "__main__":
    main(sys.argv)