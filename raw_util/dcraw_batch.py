import subprocess
import sys
import os


def convert_cr_to_tiff(basepath):
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        if d.lower().endswith(".cr2"):
                os.system(f"dcraw -v -w -n {0} -M -H {0} -o {0} -W -q {3} -4 -T {p}")

def change_dir(basepath):
    for d in os.listdir(basepath):
        p = os.path.join(basepath, d)
        if p[-4:] == "tiff":
            os.rename(os.path.join(basepath, d), os.path.join(basepath, "tiffs/" + d))

if __name__ == "__main__":
    basepath = sys.argv[1]
    convert_cr_to_tiff(basepath)
#     change_dir(basepath)


