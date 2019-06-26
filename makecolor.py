import cvtcolor as cvt
import numpy as np 
import os
import sys
import vec_noise
import sys
import matplotlib.pyplot as plt
import imageio

# appends path to custom function
sys.path.insert(0, './fractal/src')
sys.path.insert(0, './stripes/src')

import fractal
import singlegen

def main(argv):
    if len(argv) < 2:
        print('[type of map] [width] [height]')
        return 

    if len(argv) > 2:
        width = int(argv[2])
        height = int(argv[3])
    else:
        width = 2400
        height = 2400

    for i in range(2):
        # image = None
        # generate greyscale first
        if argv[1] == 'random':
            image = np.random.rand(width, height)
        elif argv[1] == 'perlin':
            image = fractal.generate_perlin_noise_2d((width, height), (8,8))
        elif argv[1] == 'fractal':
            image = fractal.generate_fractal_noise_2d((width, height), (8,8))
        elif argv[1] == 'stripe':

            # generate parameters for stripes
            image = singlegen.stripe((width, height))
            # print(image)

            # plt.imshow(image, cmap='gray')
            # plt.show()


        else:
            print('invalid type of map')
            return
        
        
        amb, direct, image = cvt.addIlluminationColor(image)

        # image = cvt.normalizeImage(image)
        # amb =  amb.astype(np.uint8)
        # direct = cvt.normalizeImage(direct)
        # print(image)
        # plt.imshow(image, cmap='Greys')
        # plt.show()

        # imageio.imwrite('./test/' + argv[1] + '_{0:05d}.png'.format(i), image)
        # imageio.imwrite('./test/' + argv[1] + '_{0:05d}.png'.format(i) + '_amb', amb)
        # imageio.imwrite('./test/' + argv[1] + '_{0:05d}.png'.format(i) + '_dir', direct)
        image = image.astype(np.float32) / 255
        np.save('./imaps/test', image)
        
        # imageio.imwrite('./test/{}_{}.png'.format(argv[1], i), image)
        # imageio.imwrite('./test/{}_{}_amb.png'.format(argv[1], i), amb)
        # imageio.imwrite('./test/{}_{}_dir.png'.format(argv[1], i), direct)

if __name__ == "__main__":
    main(sys.argv)