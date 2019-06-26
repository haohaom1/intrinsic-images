

import cv2 
import numpy as np 
import colorsys
import os 
from tqdm import tqdm


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

images = os.listdir("../Dataset_Original/Dataset_JPEG/")

for image in tqdm(images): 

    img = cv2.imread("../Dataset_Original/Dataset_JPEG/" + image)

    # print(img[20, 10, :])

    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img = img * colorsys.hsv_to_rgb(0.8, 0.9, 1.0)
    img = img.astype(np.uint8)

    cv2.imwrite("../Dataset_Original/Dataset_modified/" + image,img)

    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()