# Allen Ma 2019

import numpy as np 
import scipy.ndimage
import sys
import pyexr


def ppm_to_numpy(fname):
    ''' reads in a ppm image and returns a numpy array'''
    if fname[-3:] != "ppm":
        raise("Image {} is not ppm format".format(fname))
    else:
        return scipy.ndimage.imread(fname)

def numpy_to_exr(img, out_name):
    ''' converts numpy arr to exr file 
        img - numpy arr
        out_name - the output filename
    '''
    imageio.imwrite(f'{out_name}.exr', img)



def main(argv):
    max_val = 255
    if len(argv[1:]) < 1:
        exit("Usage: python3 ppmtonumpytoexr.py fname.ppm")
    fname = argv[1]
    out_name = fname[:-4]
    img = ppm_to_numpy(fname)
    img = img.astype(np.float32)
    img /= max_val
    print(img)
    assert(np.max(img) <= 1.0 and np.min(img) >= 0.0)
    
    shape = img.shape
    print(shape)
    pyexr.write(f"{out_name}.exr", img)



if __name__ == "__main__":
    main(sys.argv)


