'''
adds color to a greyscale image
assumes that the greyscale image is between 0-1 floating point numbers

Saves everything as np arrays

NOTE:
NAMING CONVENTION: ./data/imap_npy[amb/direct]]/[train, test]/[gen_type]%d.npy
Saves everything to train by default

Make sure naming convention is in sync with data_gen.py
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
from imapUtil import fractal, stripes

def main(argv):
    if len(argv) < 2:
        print('[type of map] [number] [train/test] [width] [height] [basepath]')
        return 

    if len(argv) > 5:
        NUM_MAPS = int(argv[2])
        dir_name = argv[3]
        width = int(argv[4])
        height = int(argv[5])
        basepath = argv[6]
    else: 
        # default values
        NUM_MAPS = 1200
        dir_name = 'train'
        width = 512
        height = 512
        basepath = "."

    print(f'{dir_name}: generating {NUM_MAPS} {argv[1]} maps {height} by {width}')

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
            image = stripes.stripe((width, height))
            # print(image)

            # plt.imshow(image, cmap='gray')
            # plt.show()


        else:
            print('invalid type of map')
            return
        
        
        amb, direct, image = cvt.addIlluminationColor(image)

        # convert to float32
        image = image.astype(np.float32)
        amb = amb.astype(np.float32)
        direct = direct.astype(np.float32)

        image_path = os.path.join(basepath, f"data/imap/imap_npy/{dir_name}/{gen_type}{i}")
        ambient_path = os.path.join(basepath, f"data/imap/imap_npy_ambient/{dir_name}/{gen_type}{i}")
        direct_path = os.path.join(basepath, f"data/imap/imap_npy_direct/{dir_name}/{gen_type}{i}")

        np.save(image_path, image)
        np.save(ambient_path, amb)
        np.save(direct_path, direct)

        print(f'saved {gen_type}{i} into {image_path} (and amb, direct)')

if __name__ == "__main__":
    main(sys.argv)