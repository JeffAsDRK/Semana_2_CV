from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2,InterpolationMode
from PIL import Image
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
DATASET_PATH = "dense_data"
DEFAULT_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
AUMENTATION_TRANSFORM = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.6),
        v2.RandomRotation(45),
    ]
)
MASK_TRANSFORM = v2.Compose([
    v2.Resize(size=(256, 256)),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.int64),
])


import time
class Proces_Data(Dataset):
    def __init__(
        self,
        df,
        transform=DEFAULT_TRANSFORM,
        path=DATASET_PATH,
        aumentation=False,
    ) -> None:

        self.df_class = df
        self.path = path
        self.transform = transform
        self.aument = aumentation

    def __len__(self):
        return len(self.df_class)

    def __getitem__(self, idx):
        img_string = f"./{self.path}/{self.df_class['name'].iloc[idx]}/frame/{self.df_class['frame'].iloc[idx]}"
        mask_string = f"./{self.path}/{self.df_class['name'].iloc[idx]}/combined/{self.df_class['combined'].iloc[idx]}"

        img = Image.open(img_string).convert("RGB")
        mask = Image.open(mask_string).convert("RGB")
        img.load()
        mask.load()
        img, mask = v2.ToImage()(img, mask)
        if self.aument:
            img, mask = AUMENTATION_TRANSFORM(img, mask)
        img = self.transform(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)
        mask=MASK_TRANSFORM(mask)
        mask=mask[0]
        return img, mask


class Load_Dataset:
    def __init__(
        self,
        df: pd.DataFrame = None,
        default_path: str = None,
        val_size: float = 0.25,
        test_size: float = 0.25,
        random_state: int = 42,
        num_workers: int = 0,
        batch_size: int = 64,
        aumentation: int = 1,
    ):
        #### inicio de funcion ####
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["name"],
        )
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_size,
            random_state=random_state,
            stratify=df_train["name"],
        )
        self.aumen = aumentation
        self.path = default_path
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.num_workers = num_workers
        self.batch_size = batch_size

    def load_train(self, transform):
        if self.aumen > 1:
            self.df_train = pd.concat([self.df_train] * self.aumen, ignore_index=True)
            data_train = Proces_Data(
                self.df_train, transform, self.path, aumentation=True
            )
        else:
            data_train = Proces_Data(self.df_train, transform, self.path)
        return DataLoader(
            data_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

    def load_val(self, transform):
        data_val = Proces_Data(self.df_val, transform, self.path)
        return DataLoader(
            data_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )

    def load_test(self, transform):
        data_test = Proces_Data(self.df_test, transform, self.path)
        return DataLoader(
            data_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )
