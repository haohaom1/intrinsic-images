# assuming 100% of images are in basepath
# this script randomly allots 20% of the data into outpath
# leaving 80% in basepath
import os, os.path
import sys
import random


def rename_paths(basepath, orig_string, replace_string=""):
    ''' renames paths from basepath/name (containing) to basepath/name (replace string) - 
        essentially does name.replace(orig_string, replace_string)
    '''
    for d in os.listdir(basepath):
        orig_name = os.path.join(basepath, d)
        new_component = d.replace(orig_string, replace_string)
        new_name = os.path.join(basepath, new_component)
        os.rename(orig_name, new_name)
        print(f"moving {orig_name} --> {new_name}")


def train_test_split(basepath, outpath, ratio=0.8):
    '''
        ratio is the dominant ratio eg. 80 in 80/20
    '''
    all_paths = []
    for d in os.listdir(basepath):
        all_paths.append(os.path.join(basepath, d))
    # randomly shuffle the paths
    random.shuffle(all_paths)
    ratio_len = int(len(all_paths) * ratio)
    train_paths = all_paths[:ratio_len]
    test_paths = all_paths[ratio_len:]

    # move the test paths
    for p in test_paths:
        component_name = os.path.split(p)[1]
        new_path_name = os.path.join(outpath, component_name)
        print(f"moving {p} --> {new_path_name}")
        os.rename(p, new_path_name)

def main(argv):
    basepath = argv[1]
    outpath = argv[2]
    train_test_split(basepath, outpath)

if __name__ == "__main__":
    main(sys.argv)