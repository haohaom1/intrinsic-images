'''
	* Maan Qraitem 
	* Colby College 
	* Create Saturated Versions of the Converted images 
'''

import cv2 
import numpy as np 
import colorsys
import os 
from tqdm import tqdm
import sys


# A fucntion that converts hsv to rgb 

def hsv2rgb(h,s,v):
	return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

kind = sys.argv[1]
images = os.listdir("../Dataset_Converted/" + kind)


'''
	* For every image with White Balance Auto and expsure 1: 
		* Create version that is saturted with Red. 
		* Save the copy in the Dataset_Converted/kind folder 

'''

for image in tqdm(images): 

	WB = image.split("_")[2]
	exp = image.split("_")[4].split(".")[0]

	if WB == "auto" and exp == "1":
		img = cv2.imread("../Dataset_Converted/" + kind + "/" + image, -1)

		# print(img[20, 10, :])

		# cv2.imshow('image',img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		img = img * colorsys.hsv_to_rgb(0.8, 0.9, 1.0)
		if kind == "linear":
			img = img.astype(np.uint16)

		else: 
			img = img.astype(np.uint8)

		cv2.imwrite("../Dataset_Converted/" + kind + "/" + image.split("_")[0] + "_" + image.split("_")[1] + "_" + 
													   image.split("_")[2] + "_" + image.split("_")[3] + "_" + 
													   image.split("_")[4] + "_" + "Red." +  image.split(".")[2],img)

		# print(img[20, 10, :])

		# quit()

		# cv2.imshow('image',img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()