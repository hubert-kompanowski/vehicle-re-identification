import pickle
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np

# dataset_path = "/home/hubert/Projects/vehicle-re-identification/data/VehicleID_V1.0/train_test_split/train_list.txt"
# images_path = "/home/hubert/Projects/vehicle-re-identification/data/VehicleID_V1.0/image"

dataset_path = "/data/VehicleID_V1.0/train_test_split/train_list.txt"
images_path = "/data/VehicleID_V1.0/image"

classes_dict = {}

data = []
with open("/data/VeRi/train_label.xml", "r") as f:
    tree = ET.fromstring(f.read())

for label in tree.find("Items"):
    data.append(
        (
            label.attrib["imageName"],
            label.attrib["vehicleID"],
        )
    )

for i, line in enumerate(data):
    if i % 1000 == 0:
        print(i)
    img_class = line[1].strip()
    if img_class not in classes_dict:
        classes_dict[img_class] = 0
    classes_dict[img_class] += 1

classes_hist = classes_dict.values()

print(min(classes_hist))
print(max(classes_hist))
print(sum(classes_hist) / len(classes_hist))

plt.figure(figsize=(10, 7))
plt.xlabel("Number of images per class")
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')

plt.hist(classes_hist, 30)  # , density=True, facecolor='g', alpha=0.75)
plt.xticks(np.arange(0, max(classes_hist), 50.0))
plt.show()
