import pickle
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np

# dataset_path = "/home/hubert/Projects/vehicle-re-identification/data/VehicleID_V1.0/train_test_split/train_list.txt"
# images_path = "/home/hubert/Projects/vehicle-re-identification/data/VehicleID_V1.0/image/"

# images_path = "/home/hubert/Projects/vehicle-re-identification/data/VeRi/image_train/"
#
# data = []
# with open("/home/hubert/Projects/vehicle-re-identification/data/VeRi/train_label.xml", "r") as f:
#     tree = ET.fromstring(f.read())
#
# for label in tree.find("Items"):
#     data.append(
#         (
#             label.attrib['imageName'],
#             label.attrib["vehicleID"],
#         )
#     )
#
# x_width = []
# y_height = []
#
# for i, line in enumerate(data):
#     if i % 1000 == 0:
#         print(i)
#     img_path = line[0].strip()
#     w, h, _ = cv2.imread(images_path + img_path).shape
#     x_width.append(w)
#     y_height.append(h)
#
# sizes = (x_width, y_height)
# pickle.dump(sizes, open("veri_sizes.p", "wb"))

# exit(0)

y_height, x_width = pickle.load(open("../veri_sizes.p", "rb"))
print(len(y_height))
x_width = np.array(x_width)
y_height = np.array(y_height)

# x_width_mask = x_width < 500
# y_height_mask = y_height < 500
#
# x_width = x_width[x_width_mask*y_height_mask]
# y_height = y_height[x_width_mask*y_height_mask]


# Creating bins
x_min = np.min(x_width)
x_max = np.max(x_width)

y_min = np.min(y_height)
y_max = np.max(y_height)
x_bins = np.linspace(x_min, x_max, 20)
y_bins = np.linspace(y_min, y_max, 20)

fig, ax = plt.subplots(figsize=(10, 7))
# Creating plot
plt.hist2d(x_width, y_height, bins=[x_bins, y_bins], cmap=plt.cm.hot)

# plt.title("Simple 2D Histogram")
plt.colorbar()
ax.set_xlabel("Image width [px]")
ax.set_ylabel("Image height [px]")

# show plot
plt.show()


x_width = np.array(x_width)
y_height = np.array(y_height)

w_id_max = x_width.argmax()
w_id_min = x_width.argmin()
h_id_max = y_height.argmax()
h_id_min = y_height.argmin()

print(x_width[w_id_max], y_height[w_id_max])
print(x_width[w_id_min], y_height[w_id_min])
print(x_width[h_id_max], y_height[h_id_max])
print(x_width[h_id_min], y_height[h_id_min])
