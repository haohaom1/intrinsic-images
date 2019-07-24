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
    os.system("python3 makecolor.py stripe 2000 test 512 512")
    os.system("python3 makecolor.py perlin 2000 test 512 512")
    os.system("python3 makecolor.py fractal 2000 test 512 512")
    os.system("python3 makecolor.py random 2000 test 512 512")

if __name__ == "__main__":
    main(sys.argv)
