import os
import sys

def main(argv):
    if len(argv) < 1:
        print("usage: specify a basepath. It should be the parent folder of the data directory")
        exit(-1)
    basepath = argv[1]
    if not os.path.isdir(basepath):
        print(f"{basepath} not a valid directory")
        exit(-1)
    os.system(f"python3 makecolor.py stripe 8000 train 512 512 {basepath}")
    os.system(f"python3 makecolor.py perlin 8000 train 512 512 {basepath}")
    os.system(f"python3 makecolor.py fractal 8000 train 512 512 {basepath}")
    os.system(f"python3 makecolor.py random 8000 train 512 512 {basepath}")

if __name__ == "__main__":
    main(sys.argv)


