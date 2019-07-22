import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
import scipy.misc
import cv2

def stripe(size):

    img_width, img_height = size

    x, y = (np.random.randint(int(img_width / 2), size=2) + int(img_width / 2))
    width = np.random.randint(int(img_width / 12 * 5)) + int(img_width / 12)

    theta = np.random.rand() * 360

    # for rotation purposes
    img_height *= 2
    img_width *= 2

    # x, y = 1400, 1900
    # width = 1000
    # theta = 30

    # print(img_width, img_height, x, y, width, theta)
    # print('y-width', y-width, 'y+width', y+width)

    data = np.ones((img_height, img_width))

    grad = np.linspace(width, 0, width) / width
    # if non linear gradient
    grad = np.square(grad) 

    grad_vals = np.repeat(grad, img_width).reshape(width, img_width)
    grad_vals = np.vstack([grad_vals, np.flip(grad_vals, axis=0)])

    # print('grad shape', grad_vals.shape)
    # print('width shape', data[y-width:y+width, :].shape)
    # print('data shape', data.shape)

    data[y-width:y+width, :] = grad_vals
    
    # # translate, rotate, translate back
    data = ndimage.shift(data, (img_height / 2 - y, img_width / 2 - x), mode='nearest')
    rot = ndimage.rotate(data, theta, mode = 'nearest')
    rot = ndimage.shift(rot, (y - img_height / 2, x - img_width / 2), mode='constant')


    # # rescale to original size
    if int(x - img_width/4) < 0 or int(y - img_height/4) < 0:
        print('error')
        return

    # print(startx, endx, starty, endy)
    startx = int(x - img_width/4)
    endx = int(x+img_width/4)
    starty = int(y - img_height/4)
    endy = int(y+img_height/4)
    rot = rot[startx:endx, starty:endy]

    # rot = rot * 255
    # rot = rot.astype(np.uint8)
    np.clip(rot, 0., 1.)

    return rot

def main(argv):
    pass


if __name__ == "__main__":
    main(sys.argv)