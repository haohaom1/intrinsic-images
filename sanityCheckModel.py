import numpy as np
import random
import json
import os
import datetime
import sys

# hardcoded
from models.janknet.janknet_separation import JankNet
from models.unet.unet_separation import UNet
from models.simpleJanknet.simple_janknet import SimpleJankNet
from models.janknet2head.janknet2head import JankNet2Head
from models.mikenet.mikenet import MikeNet
from models.strongerJanknet.strongerjanknet import StrongerJankNet
from models.brucenet.brucenet import BruceNet

# generate an image with n horizontal lines
# of size m x m
# each line of length p

def gen_horiz_line(n, m, p):
    # pick n random rows
    assert(n <= m)
    # don't want the same row
    rand_rows = np.random.choice(m, n, replace=False)
    # pick n random column starting points
    # same column is OK, so replace can be true
    rand_cols = np.random.choice(m - p + 1, n, replace=True)
    
    rand_points = np.array(list(zip(rand_rows, rand_cols)))

    # now fill the grid with 1s representing the lines
    # zeros representing the background
    grid = np.zeros([m, m])
    
    for pt in rand_points:
        grid[pt[0], pt[1]:pt[1]+p] = 1

    return grid


# generate an image with n vertical lines
# of size m x m
# each line of length p
def gen_vertical_line(n, m, p):
    # pick n random columns
    assert(n <= m)
    # don't want the same row
    rand_cols = np.random.choice(m, n, replace=False)
    # pick n random row starting points
    # same row is OK, so replace can be true
    rand_rows = np.random.choice(m - p + 1, n, replace=True)
    
    rand_points = np.array(list(zip(rand_rows, rand_cols)))

    # now fill the grid with 1s representing the lines
    # zeros representing the background
    grid = np.zeros([m, m])
    
    for pt in rand_points:
        grid[pt[0]:pt[0] + p, pt[1]] = 1

    return grid


def gen_cross_image(vert, horiz):
    assert(vert.shape == horiz.shape)
    # saturated add, cap at 1
    img = vert + horiz
    return np.clip(img, a_min=0, a_max=1)

    
# returns a batch of 64 images
def generator():
    while True:
        
        cross_data = []
        horiz_data = []
        vert_data = []
        
        for _ in range(64):
            vert = gen_vertical_line(random.randrange(20, 50), 128, random.randrange(5, 40))[:, :, np.newaxis]
            horiz = gen_horiz_line(random.randrange(20, 50), 128, random.randrange(5, 40))[:, :, np.newaxis]
            cross = gen_cross_image(vert, horiz)
            horiz_data.append(horiz)
            vert_data.append(vert)
            cross_data.append(cross)
            
        cross_data = np.array(cross_data)
        horiz_data = np.array(horiz_data)
        vert_data = np.array(vert_data)
            
        yield cross_data, (horiz_data, vert_data)


def main(argv):

    if len(argv) < 2:
        print('[name of model]')
        return

    model_name = argv[1]

    input_size = (128, 128, 3)

    # determines model name
    if model_name == "janknet":
        net = JankNet(input_size=input_size)
    elif model_name == 'unet':
        net = UNet(input_size=input_size)
    elif model_name == 'simpleJanknet':
        net = SimpleJankNet(input_size=input_size)
    elif model_name == 'janknet2head':
        net = JankNet2Head(input_size=input_size)
    elif model_name == 'mikenet':
        net = MikeNet(input_size=input_size)
    elif model_name == "strongerJanknet":
        net = StrongerJankNet(input_size=input_size)
    elif model_name == "brucenet":
        net = BruceNet(input_size=input_size)
    else:
        print(f"model name {model_name} not found")
        exit(-1)

    model = net.model
    gen = generator()

    curtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    new_dir = f'./models/{model_name}/instance_{curtime}'
    hist_path = f'{model_name}_hist'

    history_obj = model.fit_generator(gen, steps_per_epoch= 500, epochs=30, verbose=1)
    json.dump(history_obj.history, open(os.path.join(new_dir, hist_path + "_" + curtime), "w"))
    final_epoch_fpath = os.path.join(new_dir, f"final_epoch_weights_{curtime}.hdf5")
    print(f"saving model to {final_epoch_fpath}")
    net.model.save(final_epoch_fpath)


if __name__ == "__main__":
    main(sys.argv)