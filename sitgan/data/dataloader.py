import os, pdb, pickle
osp=os.path
import numpy as np
import torch
from monai.data import (
    DataLoader as monai_DL,
    CacheDataset,
)

from data.transforms import get_transforms, get_attr_transforms

def get_datalists(args):
    # should return a tuple (train_datalist, val_datalist)
    # each datalist should be a list of datapoints
    # each datapoint should be a dictionary with the following 2 keys:
    # - image: a file path to your image. can be any format handled by monai.transforms.LoadImage()
    # - attributes: a list of conditional attributes associated with the image. mask missing attributes with np.nan
    raise NotImplementedError("You need to implement this function for your own data")

def get_dataloaders(args, attr_loaders=True):
    data_settings = args["data loading"]
    train_datalist, val_datalist = get_datalists(args)
    train_transforms, val_transforms = get_transforms(args)
    train_ds = CacheDataset(
        data=train_datalist, transform=train_transforms,
        cache_num=12, cache_rate=1.0, num_workers=8,
    )
    train_loader = monai_DL(
        train_ds, batch_size=data_settings["batch size"], shuffle=True, num_workers=8, pin_memory=True,
    )

    val_ds = CacheDataset(
        data=val_datalist, transform=val_transforms,
        cache_num=6, cache_rate=1.0, num_workers=4,
    )
    val_loader = monai_DL(
        val_ds, batch_size=data_settings["batch size"], shuffle=False, num_workers=4, pin_memory=True
    )
    loaders = {"train":train_loader, "val":val_loader}
    if attr_loaders:
        transform = get_attr_transforms()
        train_ds = CacheDataset(
            data=[np.expand_dims(dp["attributes"], 0) for dp in train_datalist], transform=transform, cache_num=12, cache_rate=1.0, num_workers=1,
        )
        train_attr_loader = monai_DL(
            train_ds, batch_size=data_settings["batch size"], shuffle=True, num_workers=1, pin_memory=True,
        )
        val_ds = CacheDataset(
            data=[np.expand_dims(dp["attributes"], 0) for dp in val_datalist], transform=transform, cache_num=12, cache_rate=1.0, num_workers=1,
        )
        val_attr_loader = monai_DL(
            val_ds, batch_size=data_settings["batch size"], shuffle=True, num_workers=1, pin_memory=True
        )
        loaders = {**loaders, "train_attr":train_attr_loader, "val_attr":val_attr_loader}
    return loaders
