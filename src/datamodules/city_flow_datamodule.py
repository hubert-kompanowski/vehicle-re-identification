import xml.etree.ElementTree as ET
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class CityFlowDataset(Dataset):
    def __init__(self, df, all_ids, test=False, transform=None, aug=None):
        self.df = df
        self.all_ids = all_ids
        self.transform = transform
        self.test = test
        self.aug = aug

    def __len__(self):
        if self.test:
            return len(self.df)
        return len(self.all_ids)

    @staticmethod
    def load_image(img_path):
        with Image.open(img_path) as img:
            img = np.asarray(img, dtype=np.uint8)
            img = cv2.resize(img, (256, 256))
            # img = np.transpose(img, (2,0,1))
            return img

    def load_images(self, img_paths):
        return [self.load_image(p) for p in img_paths]

    def __getitem__(self, item):
        if self.test:
            image_path, vid = self.df.iloc[item]
            image = self.load_image(image_path)
            image = self.preprocessing(image, test=True)
            return image, int(vid)

        anchor_vehicle_id = self.all_ids[item]

        data = self.df[self.df["vehicle_id"] == anchor_vehicle_id].sample(n=2)

        anchor_image_path, v1 = data.iloc[0]
        positive_image_path, v2 = data.iloc[1]
        negative_image_path, v3 = (
            self.df[self.df["vehicle_id"] != anchor_vehicle_id].sample(n=1).iloc[0]
        )

        images_list = self.load_images(
            [anchor_image_path, positive_image_path, negative_image_path]
        )

        if self.aug is not None and self.aug.apply:
            images_list = [self.augment(image) for image in images_list]

        images = np.stack(images_list)
        images = self.preprocessing(images)
        return images, 1

    @staticmethod
    def preprocessing(images, test=False):
        if test:
            images = np.transpose(images, (2, 0, 1))
        else:
            # images: [3,H,W,C] -> [3,C,H,W]
            images = np.transpose(images, (0, 3, 1, 2))
        images = (images / 255.0).astype(np.float)
        return images

    @staticmethod
    def undo_preprocessing(images):
        # images: [3,C,H,W] -> [3,H,W,C]
        images = (images * 255.0).astype(np.uint8)
        images = np.transpose(images, (0, 2, 3, 1))
        return images

    def augment(self, images):
        transform = []

        if "horizontal_flip" in self.aug and self.aug.horizontal_flip:
            transform.append(A.HorizontalFlip(p=0.5))
        if "random_brightness_contrast" in self.aug and self.aug.random_brightness_contrast:
            transform.append(A.RandomBrightnessContrast(p=0.3))
        if "shift_scale_rotate" in self.aug and self.aug.shift_scale_rotate:
            transform.append(A.ShiftScaleRotate(p=0.5))
        if "blur" in self.aug and self.aug.blur:
            transform.append(A.GaussianBlur(p=0.5))
        if "cutout" in self.aug and self.aug.cutout:
            transform.append(A.Cutout(p=0.4, max_h_size=25, max_w_size=25))

        transform = A.Compose(transform)
        results = transform(image=images)["image"]
        return results


class CityFlowDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_fold=None,
        aug: dict = None,
        stage=None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
            ]
        )
        self.val_fold = val_fold
        self.aug = aug
        self.stage = stage

        if self.val_fold is not None:
            self.val_fold = self.val_fold[0]

        if self.stage is not None and self.stage == "first":
            image_dir_name = "sys_image_train"
            data_dir = data_dir + "_Simulation"
        else:
            image_dir_name = "image_train"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        with open(f"{to_absolute_path(data_dir)}/train_label.xml", "r") as f:
            tree = ET.fromstring(f.read())

        data = []
        for label in tree.find("Items"):
            data.append(
                (
                    f"{to_absolute_path(data_dir)}/{image_dir_name}/{label.attrib['imageName']}",
                    label.attrib["vehicleID"],
                )
            )
        self.df = pd.DataFrame(data, columns=["image_path", "vehicle_id"])

        if self.stage is not None and self.stage == "first":
            self.all_ids = self.df["vehicle_id"].unique()[:200]
        else:
            self.all_ids = self.df["vehicle_id"].unique()

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            if self.val_fold is not None:
                fold1, fold3 = train_test_split(self.all_ids, test_size=0.5, random_state=123)
                fold1, fold2 = train_test_split(fold1, test_size=0.5, random_state=123)
                fold3, fold4 = train_test_split(fold3, test_size=0.5, random_state=123)
                folds = [fold1, fold2, fold3, fold4]
                train_ids = np.array(
                    folds[0 : self.val_fold] + folds[self.val_fold + 1 :]
                ).flatten()
                val_ids = np.array(folds[self.val_fold]).flatten()
                test_ids = val_ids
            else:

                train_ids, test_ids = train_test_split(
                    self.all_ids, test_size=0.3, random_state=42
                )
                val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

            self.data_train = CityFlowDataset(
                self.df[self.df["vehicle_id"].isin(train_ids)],
                train_ids,
                test=False,
                transform=self.transforms,
                aug=self.aug,
            )
            self.data_val = CityFlowDataset(
                self.df[self.df["vehicle_id"].isin(val_ids)],
                val_ids,
                test=False,
                transform=self.transforms,
            )
            self.data_test = CityFlowDataset(
                self.df[self.df["vehicle_id"].isin(test_ids)],
                test_ids,
                test=True,
                transform=self.transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def display(self, subset="val"):
        self.setup()
        datasets = {
            "train": self.data_train,
            "val": self.data_val,
        }
        for images, _ in datasets[subset]:

            anchor_img, positive_img, negative_img = datasets[subset].undo_preprocessing(images)
            vis = np.concatenate(
                (
                    cv2.cvtColor(anchor_img, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(positive_img, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(negative_img, cv2.COLOR_RGB2BGR),
                ),
                axis=1,
            )
            cv2.imshow("win", vis)

            if cv2.waitKey(1500) == ord("q"):
                break

    def show_ids(self):
        self.setup()
        print("train", sorted(self.data_train.all_ids))
        print("val", sorted(self.data_val.all_ids))
        print("test", sorted(self.data_test.all_ids))
