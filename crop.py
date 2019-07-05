import numpy as np 
from scipy import ndimage
import random
import cv2
import argparse

'''
Used to crop the 16 bit .tiff files in various, random ways
'''

def crop(img, output_size=512):

    '''
    img is a 3 channel rgb np array 
    '''

    # if the image is in 8 bit rgb, convert to floating point
    if img.dtype == 'uint8':
        img = img / 255.

    # center crop
    d = int(output_size / 2)
    h = np.random.randint(d) + int(img.shape[0] / 2)
    w = np.random.randint(d) + int(img.shape[1] / 2)

    crop1 = img[h-d:h+d, w-d:w+d]

    # scaled crop
    max_scale = int(min(img.shape[:2]) / output_size) # maximum shrinkage to ensure a output_size is still possible
    scale = np.random.uniform() * (max_scale - 1) + 1
    img_scaled = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)), interpolation=cv2.INTER_AREA)

    crop2 = random_crop(img_scaled, output_size)

    # scaled and rotated crop
    # rotations by 90 degrees only
    angle = random.randint(1,3) * 90
    img_rot = ndimage.rotate(img, angle)

    max_scale = int(min(img_rot.shape[:2]) / output_size) # maximum shrinkage to ensure a output_size is still possible
    scale = np.random.uniform() * (max_scale - 1) + 1
    img_scaled = cv2.resize(img_rot, (int(img_rot.shape[1] / scale), int(img_rot.shape[0] / scale)), interpolation=cv2.INTER_AREA)

    crop3 = random_crop(img_scaled, output_size)
    
    flip = random.random() > 0.5
    if flip:
        direction = random.randint(0, 1)
        crop3 = cv2.flip(crop3, direction)

    # takes the largest center crop, and scales it down
    w = min(img.shape[:2]) // 2
    img_center = img[img.shape[0] // 2 - w:img.shape[0] // 2 + w, img.shape[1] // 2 - w:img.shape[1] // 2 + w]
    crop4 = cv2.resize(img_center, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # clip the crops to [0, 1]
    crop1 = np.clip(crop1, 0, 1)
    crop2 = np.clip(crop2, 0, 1)
    crop3 = np.clip(crop3, 0, 1)
    crop4 = np.clip(crop4, 0, 1)

    return crop1, crop2, crop3, crop4

def random_crop(img, output_size):
    '''
    Helper function that takes a image, and randomly crops a square image of output_size
    returns the numpy slice accordingly
    '''

    height, width, _ = img.shape
    d = int(output_size / 2)

    # h, w is the center of the crop square
    h = np.random.randint(height - 2 * output_size) + output_size
    w = np.random.randint(width - 2 * output_size) + output_size

    return img[h-d:h+d, w-d:w+d]

