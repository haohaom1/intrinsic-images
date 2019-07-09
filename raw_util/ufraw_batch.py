# similar to dcraw_batch, but uses ufraw

import subprocess
import sys
import os


def convert_cr_to_tiff(basepath):
    files = []
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        if d.lower().endswith(".cr2"):
                files.append(p)
    all_files = " ".join(f for f in files)
    os.system(f"ufraw-batch --wb=camera --base-curve=linear --restore=lch --clip=digital --linearity=1.0 --saturation=1.0 --exposure=0.0 --wavelet-denoising-threshold=0.0 --hotpixel-sensitivity=0.0 --black-point=0 --interpolation=ahd --shrink=1 --out-type=tiff --out-depth=16  --create-id=no --noexif --nozip {all_files}")

def change_dir(basepath, outdir):
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        if p.lower().endswith(".tiff") or p.lower().endswith(".tif"):
            old_path = os.path.join(basepath, d)
            new_path = os.path.join(outdir, d)
            os.rename(old_path, new_path)

def main(argv):
    '''
        takes in two absolute directories eg. basepath/* of images (asterisk means all the images under images/), 
        converts the cr2s to tiff, then saves them under outdir/*
    '''
    basepath = argv[1]
    outdir = argv[2]
    convert_cr_to_tiff(basepath)
    change_dir(basepath, outdir)

if __name__ == "__main__":
    main(sys.argv)


