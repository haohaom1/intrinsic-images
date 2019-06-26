'''
    * Maan Qraitem
    * Colby College
    * Converts each LINEAR image to different exposure and white balance settings. 
'''

import os
import subprocess
import shutil
from tqdm import tqdm

multi = {"daylight" : "2064.000000 1024.000000 1524.000000 1024.000000",
         "cloudy"   : "2222.000000 1024.000000 1382.000000 1024.000000",
         "Tungsten" : "1712.000000 1219.000000 2885.000000 1219.000000",
         "Flourescent": "1926.000000 1090.000000 2376.000000 1090.000000"}

expo = [0.05, 0.1, 0.125, 0.25, 0.5, 1.0]

def read_in_here(file):
    data = file.read().split(",")
    return data

basic = "/storage/mqrait20/linear_images"
in_here = open(os.path.join(basic, "Dataset_Converted", "in_here.txt"), "r+")
added = read_in_here(in_here)
in_here.close()

all_files = [name.split(".")[0] for name in os.listdir(os.path.join(basic, "Dataset_Original", "Dataset_JPEG"))]
to_be_added = [name for name in all_files if name not in added]

for name in tqdm(to_be_added):
    in_here = open(os.path.join(basic, "Dataset_Converted", "in_here.txt"), "a")
    for exp in expo:


        subprocess.call(["../dcraw/dcraw", "-b", str(exp), "-w", "-n", "0", "-M", "-H", "0", "-o", "0", "-W", "-q", "3", "-4", "-T", "../Dataset_Original/Dataset_RAW/" + name + ".CR2"])
        shutil.move(os.path.join(basic, "Dataset_Original", "Dataset_RAW", name + ".tiff"), os.path.join(basic, "Dataset_Converted", "linear", name + "_auto_" + "exp_" + str(exp) + ".tiff"))

        subprocess.call(["../dcraw/dcraw", "-b", str(exp), "-w", "-n", "0", "-M", "-H", "0", "-W", "-q" ,"3","-T", "-g", "2.4", "12.92", "../Dataset_Original/Dataset_RAW/" + name + ".CR2"])
        shutil.move(os.path.join(basic, "Dataset_Original", "Dataset_RAW", name + ".tiff"), os.path.join(basic,  "Dataset_Converted",  "srgb", name + "_auto_" + "exp_" + str(exp) + ".tiff"))

        for mult in multi:

            subprocess.call(["../dcraw/dcraw", "-r", multi[mult].split(" ")[0],multi[mult].split(" ")[1],multi[mult].split(" ")[2],multi[mult].split(" ")[3], "-b", str(exp), "-n", "0", "-M", "-H", "0", "-o", "0", "-W", "-q", "3", "-4", "-T", "../Dataset_Original/Dataset_RAW/" + name + ".CR2"])
            shutil.move(os.path.join(basic, "Dataset_Original", "Dataset_RAW", name + ".tiff"), os.path.join(basic,  "Dataset_Converted",  "linear", name + "_" + mult + "_" + "exp_" + str(exp) + ".tiff"))

            subprocess.call(["../dcraw/dcraw", "-r", multi[mult].split(" ")[0],multi[mult].split(" ")[1],multi[mult].split(" ")[2],multi[mult].split(" ")[3] ,"-b", str(exp), "-n", "0", "-M", "-H", "0", "-W", "-q" ,"3","-T", "-g", "2.4", "12.92", "../Dataset_Original/Dataset_RAW/" + name + ".CR2"])
            shutil.move(os.path.join(basic, "Dataset_Original", "Dataset_RAW", name + ".tiff"), os.path.join(basic,  "Dataset_Converted",  "srgb", name + "_" + mult + "_" + "exp_" + str(exp) + ".tiff"))


    in_here.write(name + ",")
    in_here.close()
