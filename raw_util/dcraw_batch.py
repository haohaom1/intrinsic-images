import subprocess
import sys
import os

basepath = "/Users/home1/Allen/summer19/intrinsic-images/intrinsic-images/data/matmap_raw/"

def main():
    
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        os.system(f"dcraw -v -w -n {0} -M -H {0} -o {0} -W -q {3} -4 -T {p}")

def change_dir():
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        if p[-4:] == "tiff":
            os.rename(os.path.join(basepath, d), os.path.join(basepath, "tiffs/" + d))

if __name__ == "__main__":
    change_dir()


