import funcy
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import pandas as pd
import torch

from nvidia.dali.pipeline import Pipeline
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ISICDataset(Dataset):
    IMG_EXT = ("png", "jpg", "jpeg", "gif")

    P_MELANOMA = 0.02
    P_BENIGN = 0.98
    PROBS = {
        0.0: P_BENIGN,
        1.0: P_MELANOMA
    }

    def __init__(self, datapath, df_labels, transform=None, size=(224, 224)):
        self.datapath = datapath

        if isinstance(df_labels, str):
            df_labels = pd.read_csv(df_labels)

        if "target" in df_labels.columns:
            self.df_labels = df_labels[~pd.isna(df_labels["target"])]
        else:
            self.df_labels = df_labels

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
        return img

    @property
    def class_weights(self):
        weights = torch.tensor(
            [1. / self.PROBS[c] for c in self.df_labels.target],
            dtype=torch.float
        )

        return weights

    def __getitem__(self, item):
        dataitem = self.df_labels.iloc[item]

        image_name = dataitem["image_name"]
        filepath = os.path.join(self.datapath, image_name + ".jpg")

        if "target" in self.df_labels.columns:
            target_val = dataitem["target"]
        else:
            target_val = np.nan

        img = self.load_image(filepath)
        img_resize = self.resize(img)

        if self.transform is not None:
            img_transform = self.transform(img_resize)
        else:
            img_transform = img_resize

        img_tensor = self.totensor(img_transform)
        img_norm = self.normalize(img_tensor)

        return image_name, img_norm, target_val


class ExternalInputIterator:
    def __init__(self, root, datapath, batch_size):
        self.root = root
        self.batch_size = batch_size

        df = pd.read_csv(datapath)
        self.df = df.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        self.i = 0
        self.n = len(self.df)

        return self

    def __next__(self):
        inputs = []
        targets = []

        for _ in range(self.batch_size):
            row = self.df.iloc[self.i]

            filename = row["image_name"]
            filepath = os.path.join(self.root, filename + ".jpg")
            with open(filepath, "rb") as f:
                img_arr = np.frombuffer(f.read(), dtype=np.uint8)
            inputs.append(img_arr)

            if "target" in self.df_labels.columns:
                target_val = dataitem["target"]
            else:
                target_val = np.nan
            targets.append(target_val)

            self.i = (self.i + 1) % self.n

        return inputs, targets

    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size, num_threads, size=(224, 224), device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)

        self.data_iterator = data_iterator
        self.input = ops.ExternalSource()
        self.target = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        resize_x, resize_y = size
        self.res = ops.Resize(device="cuda", resize_x=resize_x, resize_y=resize_y, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        self.inputs = self.input()
        self.targets = self.target()
        inputs = self.decode(self.inputs)
        resized_inputs = self.resize(inputs)
        return (resized_inputs, self.targets)

    def iter_setup(self):
        inputs, targets = self.data_iterator.next()

        self.feed_input(self.inputs, inputs)
        self.feed_input(self.targets, targets)
