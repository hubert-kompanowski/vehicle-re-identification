import xml.etree.ElementTree as ET
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
import torchvision.transforms as T


class CityFlowDataset(Dataset):
    def __init__(self, df, all_ids, test=False, transform=None):
        self.df = df
        self.all_ids = all_ids
        self.transform = transform
        self.test = test

    def __len__(self):
        if self.test:
            return len(self.df)
        return len(self.all_ids)

    def __getitem__(self, item):
        if self.test:
            image_path, vid = self.df.iloc[item]
            image = torch.tensor(read_image(image_path), dtype=torch.float)
            if self.transform:
                image = self.transform(image)
            return image, int(vid)

        anchor_vehicle_id = self.all_ids[item]

        data = self.df[self.df["vehicle_id"] == anchor_vehicle_id].sample(n=2)

        anchor_image_path, v1 = data.iloc[0]
        positive_image_path, v2 = data.iloc[1]
        negative_image_path, v3 = (
            self.df[self.df["vehicle_id"] != anchor_vehicle_id].sample(n=1).iloc[0]
        )

        results = []
        for image_path in [anchor_image_path, positive_image_path, negative_image_path]:
            image = torch.tensor(read_image(image_path), dtype=torch.float)
            if self.transform:
                image = self.transform(image)
            results.append(image)
        return results, 1


class CityFlowDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,
            val_fold=None
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # transforms.ToPILImage()
            ]
        )
        self.val_fold = val_fold

        if self.val_fold is not None:
            self.val_fold = self.val_fold[0]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        with open(f"{to_absolute_path(data_dir)}/train_label.xml", "r") as f:
            tree = ET.fromstring(f.read())

        data = []
        for label in tree.find("Items"):
            data.append(
                (
                    f"{to_absolute_path(data_dir)}/image_train/{label.attrib['imageName']}",
                    label.attrib["vehicleID"],
                )
            )
        self.df = pd.DataFrame(data, columns=["image_path", "vehicle_id"])
        self.all_ids = self.df["vehicle_id"].unique()

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            if self.val_fold is not None:
                fold1, fold3 = train_test_split(self.all_ids, test_size=0.5, random_state=123)
                fold1, fold2 = train_test_split(fold1, test_size=0.5, random_state=123)
                fold3, fold4 = train_test_split(fold3, test_size=0.5, random_state=123)
                folds = [fold1, fold2, fold3, fold4]
                train_ids = np.array(folds[0:self.val_fold] + folds[self.val_fold + 1:]).flatten()
                val_ids = np.array(folds[self.val_fold]).flatten()
                test_ids = val_ids
            else:

                train_ids, test_ids = train_test_split(self.all_ids, test_size=0.3, random_state=42)
                val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

            self.data_train = CityFlowDataset(
                self.df[self.df["vehicle_id"].isin(train_ids)],
                train_ids,
                test=False,
                transform=self.transforms,
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
                transform=self.transforms
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

    def display(self, subset='val'):
        self.setup()
        loaders = {
            'train': self.train_dataloader(),
            'val': self.val_dataloader(),
        }
        for images, _ in loaders[subset]:
            anchor_img, positive_img, negative_img = images



            vis = np.concatenate((
                cv2.cvtColor(anchor_img.long().squeeze().permute(1, 2, 0).numpy().astype('uint8'), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(positive_img.long().squeeze().permute(1, 2, 0).numpy().astype('uint8'), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(negative_img.long().squeeze().permute(1, 2, 0).numpy().astype('uint8'), cv2.COLOR_RGB2BGR),
            ), axis=1)
            cv2.imshow('win', vis)

            if cv2.waitKey(1500) == ord('q'):
                break

    def show_ids(self):
        self.setup()
        print('train', sorted(self.data_train.all_ids))
        print('val', sorted(self.data_val.all_ids))
        print('test', sorted(self.data_test.all_ids))
