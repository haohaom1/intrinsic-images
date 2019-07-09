# Allen Ma
# This utility file serves to correct the annoyances of ufraw not preserving the image sizes
# The issue is that the ufraw_batch utility adds a padding on the 80D images
# The code also flips the images that are portrait mode because that might also be annoying


# instead of maintaining the size 

import numpy as np
import image_format_converter as converter
import scipy.ndimage
import cv2
import sys
import os

EOS_80D_correct = [4000, 6000]
EOS_80D_incorrect = [4056, 6288]

EOS_REBEL_correct = [3465, 5202]

def correct_img(img):
    diff_row = EOS_80D_incorrect[0] - EOS_80D_correct[0]
    diff_col = EOS_80D_incorrect[1] - EOS_80D_correct[1]
    img = img[diff_row:,diff_col:]
    # crop the image
    assert(list(img.shape[0:2]) == EOS_80D_correct)
    return img

def main(argv):
    '''
        basepath: tiff images are stored in basepath/*
        This function crops to the correct size, and saves them in the same format
    '''
    basepath = argv[1]
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        if d.lower().endswith(".tif") or d.lower().endswith(".tiff"):
            fname = os.path.join(basepath, d)

            print(fname)
            img = converter.tiff_to_numpy(fname)
            # check whether it's messed up

            dim = list(img.shape[0:2])
            print(dim)

            # # 80D landscape
            # if dim != EOS_80D_correct:
            #     if dim == EOS_80D_incorrect:
            #         img = correct_img(img)
            #     # 80D portrait
            #     elif dim == list(reversed(EOS_80D_incorrect)):
            #         # rotate the image
            #         img = scipy.ndimage.rotate(img, 90)
            #         img = correct_img(img)
            #     # Rebel portrait
            #     elif dim == list(reversed(EOS_REBEL_correct)):
            #         # rotate the image
            #         img = scipy.ndimage.rotate(img, 90)
            #         assert(dim == EOS_REBEL_correct)
            #     else:
            #         assert(dim == EOS_REBEL_correct)

            # # write out the image
            # cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))





if __name__ == "__main__":
    main(sys.argv)
