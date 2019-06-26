'''
    * Maan Qraitem
    * Colby College
    * Purpose: Creates a Train and Test set of the dataset and saves them as numpy arrays. 
    * Input: python create_training_2.py (kind: log/linear/srgb)
'''


import os
import xml.etree.ElementTree as etree
import shutil
import matplotlib
import cv2
import numpy as np
import random
from tqdm import tqdm
import sys

random.seed(1299)

IMG_SIZE = 512

directory = "../"

if (len(sys.argv) < 2):
    print("Not enought arugments")
    quit()

kind = sys.argv[1]

LABELS = ["sour patches kids", "sour patches watermellon", "sweedish fish 1", "sweedish fish 2", "sweedish fish 3", 
          "C3PO", "pop tarts", "Pringles", "GoBoster", "Trolli"]

def read_in_here(file):
    data = file.read().split(",")
    return data

def similiar_lists(one, two): 
    for i in one: 
        for j in two: 
            if i == j: 
                return False 
    return True

def similar_same(one): 
    for i, ele in enumerate(one): 
        for j in one[i+1:]:
            if ele == j: 
                print(ele)
                return False 
    return True


'''
    * Takes a list of original images (images), labels (list of xml files of the labels)
    * Builds a dictionary where the key is a label and the value is every image the belongs to that label. 
'''

def constrcut_labels(images, labels): 

    dict = {}

    for label in labels:
        root = etree.parse(os.path.join(directory, "labels", label))
        objects = root.findall('object')
        for object in objects:
            name = object[1].text
            if name not in dict: 
                dict[name] = [] 
                dict[name].append(label.split(".")[0])
            else: 
                dict[name].append(label.split(".")[0])


    background = [] 
    for image in images: 
        if image.split(".")[0] + ".xml" not in labels: 
            background.append(image.split(".")[0])

    dict["background"] = background[:]

    # for label in dict: 
    #     print(label, similar_same(dict[label]))

    return dict


# total = 0
# for label in dict_labels: 
#     total += len(dict_labels[label])



# sum = 0
# for i in TEST: 
#     # print(similiar_lists(TRAIN[i], TEST[i]))
#     # print(similiar_lists(TRAIN[i], VALIDATION[i]))
#     # print(similiar_lists(TEST[i], VALIDATION[i]))
#     sum += len(TRAIN[i]) 
#     sum += len(TEST[i]) 
#     sum += len(VALIDATION[i]) 
# print(sum, len(images_original))


'''
    * Takes an image name, the list of images in Dataset_converted/kind, and a check. 
    * returns a list of all the images in Dataset_Converted/kind that matches the image (img). 
    * The check is used to indicate whether to return only auto WB and full exposure only images or all the variations. 
'''

def get_matching_images(img, directory, check): 
    images = os.listdir(directory)
    final = [] 
    for im in images: 
        im_name = im.split("_")[0] + "_" + im.split("_")[1]
        WB = im.split("_")[2]
        exp = im.split("_")[4].split(".")[0]
        saturation = im.split("_")[5].split(".")[0]

        full_exp = im.split("_")[4].split(".")[0] + "." + im.split("_")[4].split(".")[1]
        if im_name == img: 
            if check == False:
                final.append([im, WB, full_exp, saturation])
            elif WB == "auto" and exp == "1" and saturation == "normal":
                final.append([im, WB, full_exp, saturation])
    return final


'''
    * Takes the list of matching images from get_matching images (images_list), the list where to put the full data: images and labels (set_final), and index 
    of the label hot vector that should be set to 1. Example: [0, 0, 0], idx = 1 ----> [0, 1, 0]
    * Adds the every image in images_list to the set final along with its correct label. 
'''

def add_matching_images(images_list, set_final, idx): 
    for img in images_list: 
        image = cv2.imread(os.path.join(directory, "Dataset_Converted", kind, img[0]), -1)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        Y = [0, 0, 0, 0]
        Y[idx] = 1
        set_final.append([np.array(image), Y[:], img[1] + "_" + img[2], img[3]])
    

'''
    * The most high level function in here that calls both add_matching_images and get_matching_images. 
    * Takes in a dictionary of labels as keys and their corresponding images that need to be added to the final list ttt_list. 
    * Basically constructs either the list of train or the list of test. 
    * Check usage is mentioned get_matching_images. 
    * Check is normally False for test and True for train (only train on Auto/1.0 exposure) and test on everything else. 
'''

def create_list(SET, check):

    ttt_list = []
    for label in SET: 
        if label == "sweedish fish 1":
            for img in SET[label]: 
                final = get_matching_images(img, os.path.join(directory, "Dataset_Converted", kind), check)
                add_matching_images(final, ttt_list, 0)

        elif label == "sweedish fish 2":
            for img in SET[label]: 
                final = get_matching_images(img, os.path.join(directory, "Dataset_Converted", kind), check)
                add_matching_images(final, ttt_list, 1)

        elif label == "sweedish fish 3":
            for img in SET[label]: 
                final = get_matching_images(img, os.path.join(directory, "Dataset_Converted", kind), check)
                add_matching_images(final, ttt_list, 2)  

        else: 
            for img in SET[label]: 
                final = get_matching_images(img, os.path.join(directory, "Dataset_Converted", kind), check)
                add_matching_images(final, ttt_list, 3)


        print("Done with label: " + label)

    print("Done")    

    return ttt_list


images_original = os.listdir(os.path.join(directory, "Dataset_Original", "Dataset_JPEG"))
random.shuffle(images_original)

labels = os.listdir(os.path.join(directory, "labels"))
images_full = os.listdir(os.path.join(directory, "Dataset_Converted", kind))


dict_labels = constrcut_labels(images_original, labels)

'''
    * Splits the dataset into 10 chuncks. 
    * Create 10 possible train/test set. 
    * Test is 1/10 of the dataset and Train is 9/10. 
    * The dataset is split in a way such that each label is split into 1/10 for testa and 9/10 for train instead of splitting the 
    whole dataset at once.
    * The resulting dictionaries are passed to create list and then the resulted list is saved as a numpy array. 
'''

for i in tqdm(range(10)): 
    
    TRAIN = {} 
    TEST = {}

    for label in dict_labels: 
        sub_images = dict_labels[label][:]
        marker = i*int(len(sub_images)/10)
        TEST[label] = sub_images[marker:marker + int(len(sub_images)/10)]
        TRAIN[label] = sub_images[:marker] + sub_images[marker + int(len(sub_images)/10):]

    
                
    train_data = np.array(create_list(TRAIN, True))
    test_data = np.array(create_list(TEST, False))


    # for img in train_data[:10]:
    #     cv2.imshow('image',img[0])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # quit(0)

    print(train_data.shape)
    print(test_data.shape)


    np.save("/storage/mqrait20/linear_images/Data/train" + "_" + kind + "_" + str(i), train_data)
    np.save("/storage/mqrait20/linear_images/Data/test" + "_" + kind + "_" + str(i), test_data)


