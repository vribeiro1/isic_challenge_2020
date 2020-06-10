import funcy
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ISICDataset(Dataset):
    IMG_EXT = ("png", "jpg", "jpeg", "gif")

    def __init__(self, datapath, df_labels, transform=None, size=(224, 224)):
        self.datapath = datapath

        if isinstance(df_labels, str):
            df_labels = pd.read_csv(df_labels)
        self.df_labels = df_labels[~pd.isna(df_labels["target"])]

        self.transform = transform
        self.resize = transforms.Resize(size)
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df_labels)

    @classmethod
    def is_image(cls, fname, extensions=None):
        if extensions is None:
            extensions = cls.IMG_EXT
        return not fname.startswith(".") and any([fname.endswith(ext) for ext in extensions])

    def load_image(self, fpath):
        img = Image.open(fpath).convert("RGB")
        if self.erase_text:
            img = erase_text_from_image(img)
        return img

    def __getitem__(self, item):
        dataitem = self.df_labels.iloc[item]

        image_name = dataitem["image_name"]
        filepath = os.path.join(self.datapath, dataitem["filepath"])

        if "target" in self.df_labels.columns:
            target_val = dataitem["target"]
        else:
            target_val = None

        img = self.load_image(filepath)
        img_resize = self.resize(img)

        if self.transform is not None:
            img_transform = self.transform(img_resize)
        else:
            img_transform = img_resize

        img_tensor = self.totensor(img_transform)
        img_norm = self.normalize(img_tensor)

        return image_name, img_norm, target_val
