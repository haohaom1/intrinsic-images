# Allen Ma 2019

import numpy as np 
import matplotlib.pyplot as plt
import sys
import pyexr
import os.path

basepath = "/Users/home1/Allen/summer19/intrinsic-images/intrinsic-images/test_data"

def ppm_to_numpy(fname):
    ''' reads in a ppm image and returns a numpy array'''
    if fname[-3:] != "ppm":
        print("Image {} is not ppm format".format(fname))
        raise(AssertionError)
    else:
        return plt.imread(fname)

def numpy_to_exr(img, out_name):
    ''' converts numpy arr to exr file 
        img - numpy arr
        out_name - the output filename
    '''
    imageio.imwrite(f'{out_name}.exr', img)

def save_to_npy(img, out_name, bpath):
    np.save(os.path.join(bpath, out_name), img)

def main(argv):
    max_val = 255
 
    basedir = os.path.join(basepath, "test_mmap_ppm")
    for d in os.listdir(basedir):
        fname = os.path.join(basedir, d)
        out_name = fname[:-4]
        if fname.endswith("ppm"):
            img = ppm_to_numpy(fname)
            img = img.astype(np.float32)
            img /= max_val
            assert(np.max(img) <= 1.0 and np.min(img) >= 0.0)
            save_to_npy(img, out_name, os.path.join(basepath, "test_mmap_npy"))


if __name__ == "__main__":
    main(sys.argv)


