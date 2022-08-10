import xml.etree.ElementTree as ET

import pandas as pd

data_dir = "../data/AIC21_Track2_ReID_Simulation/"


with open(f"{data_dir}/train_label.xml", "r") as f:
    tree = ET.fromstring(f.read())

data = []
for label in tree.find("Items"):
    data.append(
        (
            f"{data_dir}/sys_image_train/{label.attrib['imageName']}",
            label.attrib["vehicleID"],
        )
    )
df = pd.DataFrame(data, columns=["image_path", "vehicle_id"])
all_ids = df["vehicle_id"].unique()

print(all_ids[:200])
