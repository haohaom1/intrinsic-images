'''
adds color to a greyscale image
assumes that the greyscale image is between 0-1 floating point numbers

Saves everything as np arrays

NAMING CONVENTION: ./[(amb, dir, imap)]/[gen_type]%d.npy
'''


import cvtcolor as cvt
import numpy as np 
import os
import sys
import vec_noise
import sys
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
import imageio

# appends path to custom function
sys.path.insert(0, './fractal/src')
sys.path.insert(0, './stripes/src')

import fractal
import singlegen


NUM_MAPS = 1200


def main(argv):
    if len(argv) < 2:
        print('[type of map] [width] [height]')
        return 

    if len(argv) > 2:
        width = int(argv[2])
        height = int(argv[3])
    else:
        width = 512
        height = 512

    # make sure the saving directories exist
    paths = ['./imaps', './imaps/imaps', './imaps/ambient', './imaps/direct']
    for p in paths: 
        if not os.path.isdir(p):
            os.mkdir(p)

    for i in range(NUM_MAPS):
        # image = None
        # generate greyscale first

        gen_type = argv[1]

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
        # image = image.astype(np.float32) / 255.
        # amb = amb.astype(np.float32) / 255.
        # direct = direct.astype(np.float32) / 255.

        np.save(f'./imaps/final/{gen_type}{i}', image)
        np.save(f'./imaps/ambient/{gen_type}{i}', amb)
        np.save(f'./imaps/direct/{gen_type}{i}', direct)

        print('saved', gen_type, i)

if __name__ == "__main__":
    main(sys.argv)