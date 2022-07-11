import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from hydra.utils import to_absolute_path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from torchvision.transforms import transforms


class CityFlowDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path, vehicle_id = self.df.iloc[item]
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image, vehicle_id


class CityFlowDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (40000, 4000, 8000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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
            train_set = CityFlowDataset(self.hparams.data_dir, transform=self.transforms)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=train_set,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
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
