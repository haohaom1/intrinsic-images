# converts greyscale images to a random colored image

import numpy as np
import sys
import os
from matplotlib.image import imread
from PIL import Image
import scipy.misc
import cv2


def hsv2rgb(h, s, v):
    def f(n):
        k = np.mod(n + h * 6, 6)
        res = v - v*s*np.maximum(np.minimum(np.minimum(k, 4-k), np.ones(h.shape)), np.zeros(k.shape))
        return res

    # src: wikipedia
    r, g, b = f(5), f(3), f(1)
    
    return r, g, b

def addIlluminationColor(image,
    ambient_std_saturation = 0.15,
    ambient_std_intensity = 0.10,
    ambient_mean_intensity = 0.20,
    ambient_lower_bound_intensity = 0.03,

    direct_std_saturation = 0.15,
    direct_std_intensity = 0.30,
    direct_mean_intensity = 0.80,
    direct_lower_bound_intensity = 0.25,
    min_ambient_direct_ratio = 0.50):

    '''
    Assumes image is a greyscale image of (n, m) np array
    The greyscale image has intensity varying from 0-1
    '''
    
    # normalize b/w 0 and 1
    if np.sum(image > 1) > 0:
        image = image / 255.

    hue_amb = 2 * np.pi * np.random.rand() * np.ones(image.shape)
    # sat_amb = min(np.abs(np.random.randn()) * ambient_std_saturation, 1) * np.ones(image.shape)
    sat_amb = np.clip(np.ones(image.shape) * np.abs(np.random.randn()) * ambient_std_intensity, a_min=None, a_max=1)
    val_amb = max(np.abs(np.random.randn() * ambient_std_intensity + ambient_mean_intensity), ambient_lower_bound_intensity) * np.ones(image.shape)
    val_amb = np.clip(np.abs(np.random.randn()) * ambient_std_intensity + ambient_mean_intensity * np.ones(image.shape), a_min=ambient_lower_bound_intensity, a_max=None)

    hue_dir = 2 * np.pi * np.random.rand() * np.ones(image.shape)
    sat_dir = min(np.abs(np.random.randn()) * direct_std_saturation, 1) * np.ones(image.shape)
    # val_dir = np.abs(image * direct_std_intensity + direct_mean_intensity)
    # val_dir = np.maximum(np.minimum(1., val_dir), direct_lower_bound_intensity) # bounds in [lowerbound, 1]

    # bounds between 0% and 70%
    val_mult = np.abs(np.random.randn()) * direct_std_intensity + direct_mean_intensity
    val_dir = np.clip(image * val_mult, direct_lower_bound_intensity, 1.)

    assert(np.sum((val_dir > 1) | (val_dir < 0)) == 0)
    assert(np.sum((val_amb > 1) | (val_amb < 0)) == 0)

    # hue_amb *= np.ones(shape)
    # sat_amb *= np.ones(shape)
    # val_amb *= np.ones(shape)
    # hue_dir *= np.ones(shape)
    # sat_dir *= np.ones(shape)
    # val_dir *= np.ones(shape)

    # img_amb = (np.stack([hue_amb, sat_amb, val_amb], axis=2) * [180, 255, 255]).astype(np.uint8)
    # img_amb = cv2.cvtColor(img_amb, cv2.COLOR_HSV2RGB).astype(np.uint16)

    # img_dir = (np.stack([hue_dir, sat_dir, val_dir], axis=2) * [180, 255, 255]).astype(np.uint8)
    # img_dir = cv2.cvtColor(img_dir, cv2.COLOR_HSV2RGB).astype(np.uint16)

    ra, ga, ba = hsv2rgb(hue_amb, sat_amb, val_amb)
    img_amb = np.stack([ra, ga, ba], axis=2)

    rd, gd, bd = hsv2rgb(hue_dir, sat_dir, val_dir)
    img_dir = np.stack([rd, gd, bd], axis=2)

    # adds 2 np.float arrays; image: [0, 2.]
    image = img_amb + img_dir
    
    # image = image / [255, 255, 255]

    return img_amb, img_dir, image

def normalizeImage(image):
    '''
    scales a rgb image between 0-255
    '''
    image = (image - image.min()) / (image.max() - image.min())
    image *= 255
    image = image.astype(np.uint8)
    return image


def main(argv):
    if len(argv) < 3:
        print('[grey_images] [color_images]')
        return 

    gray_path = argv[1]
    color_path = argv[2]

    if not os.path.isdir(color_path):
        os.mkdir(color_path)

    for image_path in os.listdir(gray_path)[:10]:

        image = imread(os.path.join(gray_path, image_path))

        # randomly sample a hue and saturation to an existing greyscale image
        rgb_img = addIlluminationColor(image)

        # hue = np.random.rand() * np.pi * 2 * np.ones(image.shape)
        # saturation = np.ones(image.shape)
        # intensity = image / 255

        # r, g, b = hsi2rgb(hue, saturation, intensity)

        # rgb_img = np.stack([r, g, b], axis=2)

        fname = os.path.join(color_path, ('color' + image_path))
        scipy.misc.imsave(fname, rgb_img)

        print('saving', fname)

if __name__ == "__main__":
    main(sys.argv)
