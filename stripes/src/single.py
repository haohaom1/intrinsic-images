import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys

def main(argv):

    if len(argv) < 7:
        print('[height] [width] [x] [y] [width] [angle]')
        return

    img_height, img_width, x, y, width = tuple([int(x) for x in argv[1:-1]])

    theta = float(argv[-1])

    data = np.zeros((img_height, img_width))

    grad = np.linspace(0, width, width) / width
    # if non linear gradient
    # grad = np.sqrt(grad) 

    grad_vals = np.repeat(grad, img_width).reshape(width, img_width)
    grad_vals = np.vstack([grad_vals, np.flip(grad_vals, axis=0)])

    data[y-width:y+width, :] = grad_vals
    
    # translate, rotate, translate back
    data = ndimage.shift(data, (img_height / 2 - y, img_width / 2 - x), mode='nearest')
    data = ndimage.rotate(data, theta, mode = 'nearest')
    rot = ndimage.shift(data, (y - img_height / 2, x - img_width / 2), mode='constant')


    # rescale to original size
    if int(x - img_width/2) < 0:
        startx = 0
        endx = img_width
    else:
        startx = int(x - img_width/2)
        endx = int(x+img_width/2)

    if int(y - img_height/2) < 0:
        starty = 0
        endy = img_height
    else:
        starty = int(y - img_height/2)
        endy = int(y+img_height/2)

    print(startx, endx, starty, endy)

    rot = rot[startx:endx, starty:endy]

    plt.imshow(rot, cmap='binary')
    plt.show()

if __name__ == "__main__":
    main(sys.argv)