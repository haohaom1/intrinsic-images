import os

def main(argv):
    basepath = argv[1]
    if not os.path.isdir(basepath):
        print(f"{basepath} not a valid directory")
        exit(-1)
    os.system(f"python3 makecolor.py stripe 50 train 512 512 {basepath}")
    os.system(f"python3 makecolor.py perlin 50 train 512 512 {basepath}")
    os.system(f"python3 makecolor.py fractal 50 train 512 512 {basepath}")
    os.system(f"python3 makecolor.py random 50 train 512 512 {basepath}")

if __name__ == "__main__":
    main(sys.argv)


