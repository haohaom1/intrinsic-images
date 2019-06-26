import numpy as np
import vec_noise
import scipy.misc

def main():

    width, height = 2400, 2400
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    octaves = np.arange(50, 150, 3)
    freq_mult = np.array([5, 10, 15, 20])

    idx = 0
    for octa in octaves:
        for mult in freq_mult:
            freq = octa * mult 
            img = vec_noise.snoise2(x / freq, y / freq, octa) * 127.0 + 128.0
            scipy.misc.imsave('../images/perlin{}.jpg'.format(idx), img)
            idx += 1

if __name__ == "__main__":
    main()