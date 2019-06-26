import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys
import scipy.misc
import cv2

def main(argv):

    img_height, img_width = 4800, 4800 
    

    for i in range(20, 25):

        x, y = (np.random.randint(1200, size=2) + 1200)
        width = np.random.randint(1000) + 200

        theta = np.random.rand() * 360

        # x, y = 1400, 1900
        # width = 1000
        # theta = 30

        print(x, y, width, theta)

        data = np.ones((img_height, img_width))

        grad = np.linspace(width, 0, width) / width
        # if non linear gradient
        grad = np.square(grad) 

        grad_vals = np.repeat(grad, img_width).reshape(width, img_width)
        grad_vals = np.vstack([grad_vals, np.flip(grad_vals, axis=0)])

        print('grad shape', grad_vals.shape)
        print('width shape', data[y-width:y+width, :].shape)

        data[y-width:y+width, :] = grad_vals
        
        # # translate, rotate, translate back
        data = ndimage.shift(data, (img_height / 2 - y, img_width / 2 - x), mode='nearest')
        rot = ndimage.rotate(data, theta, mode = 'nearest')
        rot = ndimage.shift(rot, (y - img_height / 2, x - img_width / 2), mode='constant')


        # # rescale to original size
        if int(x - img_width/4) < 0 or int(y - img_height/4) < 0:
            print('error')
            break

        # print(startx, endx, starty, endy)
        startx = int(x - img_width/4)
        endx = int(x+img_width/4)
        starty = int(y - img_height/4)
        endy = int(y+img_height/4)
        rot = rot[startx:endx, starty:endy]

        # rot = rot[int(x - img_width/2):int(x+img_width/2), int(y-img_height/2):int(y+img_height/2)]


        scipy.misc.imsave('../images/stripe{}.jpg'.format(i), rot)
        # plt.imshow(rot, cmap='binary')
        # plt.show()

if __name__ == "__main__":
    main(sys.argv)