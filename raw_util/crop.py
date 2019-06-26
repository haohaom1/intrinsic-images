# Allen Ma Summer 2019

# material maps are in some raw format
# convert material maps into png files
# take random N x N crops of the photos

import cv2
import random
import sys
import imageio
import rawpy

def crop(sz, img, prefix=None):
    '''
        sz - size of the crops
        n - number of photos desired (if n is too large there may be overlap)
        img - cv2 mat type
    '''
    # generate a valid boundary

    h, w = img.shape[:2]

    rand_y = random.randint(0, h - sz)
    rand_x = random.randint(0, w - sz)

    print(rand_y)
    print(rand_x)

    crop_img = img[rand_y:rand_y+sz, rand_x:rand_x+sz]

    # save the image
    if not prefix:
        cv2.imwrite("frame.png", crop_img)
    else:
        cv2.imwrite(f"{prefix}.png", crop_img)


def main(argv):
    args = argv[1:]
    if len(args) < 3:
        print("Usage: python3 crop.py size num_images filename [prefix]")
        exit()

    prefix = None
    
    if len(args) == 3:
        sz = int(args[0])
        n = int(args[1])
        fname = args[2]
    elif len(args) == 4:
        sz = int(args[0])
        n = int(args[1])
        fname = args[2]
        prefix = args[3]
    else:
        print("Error: Too many arguments")
        exit()

    with rawpy.imread(fname) as raw:
        img = raw.postprocess()

    crop(sz, img, prefix)




if __name__ == "__main__":
    main(sys.argv)
