'''
    * Maan Qraitem
    * Colby College
    * Linear Images research.
    * Purpose: Use the labels in LABELS folder to produce a count sheet of each category.
'''



import os
import xml.etree.ElementTree as etree
import shutil
import matplotlib.pyplot as plt

directory = "../"

Raw = []
Jpeg = []
labels = []

for file in os.listdir(os.path.join(directory, "Dataset_Original", "Dataset_RAW")):
    Raw.append(file)

for file in os.listdir(os.path.join(directory, "Dataset_Original", "Dataset_JPEG")):
    Jpeg.append(file)

for file in os.listdir(os.path.join(directory, "labels")):
    labels.append(file)


names = []
all_names = []

for label in labels:

    root = etree.parse(os.path.join(directory, "labels", label))
    objects = root.findall('object')
    for object in objects:
        all_names.append(object[1].text)
        # if (object[1].text == "sour patches kids"):
        #     print(label)
        if object[1].text not in names:
            names.append(object[1].text)


X = []
Y = []


for name in names:
    count = 0
    for all in all_names:
        if all == name:
            count += 1
    X.append(name)
    Y.append(count)



X.append("background")
Y.append(len(Jpeg)-len(labels))

for i, name in enumerate(X):
    print(name, ": ", Y[i])
    print()


print("Total is: ", sum(Y))

quit()


y_pos = range(len(names) + 1)

plt.bar(y_pos, Y, align='center', alpha=0.5)
plt.xticks(y_pos, X)
plt.ylabel('Occurences')
plt.title('Candy classes')

plt.show()
