import numpy as np 
from scipy import ndimage
import random
import cv2
import argparse
import math
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

    # CROP1: center crop
    d = int(output_size / 2)
    h = np.random.randint(d) + int(img.shape[0] / 2)
    w = np.random.randint(d) + int(img.shape[1] / 2)

    crop1 = img[h-d:h+d, w-d:w+d]

    # CROP2: scaled crop
    max_scale = int(min(img.shape[:2]) / output_size) # maximum shrinkage to ensure a output_size is still possible
    scale = np.random.uniform() * (max_scale - 1) + 1
    img_scaled = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)), interpolation=cv2.INTER_AREA)

    crop2 = random_crop(img_scaled, output_size)

    # CROP3: scaled and rotated crop
    angle = random.random() * 360.
    img_rotated = ndimage.rotate(img, angle)
    img_rotated_cropped = crop_around_center(
        img_rotated,
        *largest_rotated_rect(
            img.shape[1],
            img.shape[0],
            math.radians(angle)
        )
    )

    max_scale = int(min(img_rotated_cropped.shape[:2]) / output_size) # maximum shrinkage to ensure a output_size is still possible
    scale = np.random.uniform() * (max_scale - 1) + 1
    img_scaled = cv2.resize(img_rotated_cropped, (int(img_rotated_cropped.shape[1] / scale), int(img_rotated_cropped.shape[0] / scale)), interpolation=cv2.INTER_AREA)

    crop3 = random_crop(img_scaled, output_size)
    
    flip = random.random() > 0.5
    if flip:
        direction = random.randint(0, 1)
        crop3 = cv2.flip(crop3, direction)

    # CROP4: takes the largest center crop, and scales it down
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
    assert(height >= output_size and width >= output_size)

    d = output_size // 2

    # print(height, width, output_size)

    # ch, cw is the center of the crop square
    ch = np.random.randint(height - output_size) + output_size // 2
    cw = np.random.randint(width - output_size) + output_size // 2

    return img[ch-d:ch+d, cw-d:cw+d]


### helper functions that crop a rotated image
### source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

