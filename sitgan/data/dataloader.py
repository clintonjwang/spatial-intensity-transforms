import os, pdb
osp=os.path
from glob import glob
import dill as pickle
import os, yaml
osp = os.path
import numpy as np

import numpy as np
import nibabel as nib

import torch
from torchvision import transforms
from monai.data import (
    DataLoader as monai_DL,
    CacheDataset,
)

import args as args_module
from data.transforms import get_transforms, get_attr_transforms
ANALYSIS_DIR = osp.expanduser("~/code/sitgan/temp")

def get_dataloaders_for_dataset(dataset, batch_size, augment=True, attr_loaders=False):
    if dataset == "2D MRI-GENIE FLAIR":
        args = args_module.args_from_file(osp.expanduser("~/code/sitgan/configs/mrigenie.yaml"))
    elif dataset == "2D ADNI T1":
        args = args_module.args_from_file(osp.expanduser("~/code/sitgan/configs/adni.yaml"))
    if augment is False:
        args["augmentations"] = None
    args["data loading"]["batch size"] = batch_size

    return get_dataloaders(args, attr_loaders=attr_loaders)

def get_dataloaders(args, attr_loaders=True, overwrite=False):
    paths = args["paths"]
    data_settings = args["data loading"]
    attrs = data_settings["attributes"]
    data_root = paths["data root"]

    if args["dataset"] == "3D MRI-GENIE FLAIR":
        gasros_dir = osp.join(data_root, paths["GASROS"]["3D npy subdir"])
        gasros_datalist = produce_datalist(gasros_dir, attrs)
        mrig_dir = osp.join(data_root, paths["MRI-GENIE"]["3D npy subdir"])
        train_datalist, val_datalist = produce_datalist_mrigenie(mrig_dir, attrs)
        train_datalist += gasros_datalist
        normalizations = normalize_datalist(train_datalist)
        normalize_datalist(val_datalist, normalizations)

    elif args["dataset"] == "2D MRI-GENIE FLAIR":
        path = osp.join(ANALYSIS_DIR, "2d_mrigenie.dat")
        if osp.exists(path) and overwrite is False:
            train_datalist, val_datalist = pickle.load(open(path, "rb"))
        else:
            gasros_dir = osp.join(data_root, paths["GASROS"]["2D npy subdir"])
            gasros_datalist = produce_datalist(gasros_dir, attrs)
            mrig_dir = osp.join(data_root, paths["MRI-GENIE"]["2D npy subdir"])
            train_datalist, val_datalist = produce_datalist_mrigenie(mrig_dir, attrs)
            train_datalist += gasros_datalist
            normalizations = normalize_datalist(train_datalist)
            normalize_datalist(val_datalist, normalizations)
            np.save(osp.join(ANALYSIS_DIR, "mrigenie_normalizations.npy"), normalizations)
            pickle.dump((train_datalist, val_datalist), open(path, "wb"))

    elif args["dataset"] == "3D ADNI T1":
        data_dir = osp.join(data_root, paths["3D T1 npy subdir"])
        datalist = produce_datalist(data_dir, attrs)
        train_datalist, val_datalist = split_datalist(datalist, train_fraction=args["data loading"]["train fraction"])
        normalizations = normalize_datalist(train_datalist)
        normalize_datalist(val_datalist, normalizations)

    elif args["dataset"] == "2D ADNI T1":
        path = osp.join(ANALYSIS_DIR, "2d_adni.dat")
        if osp.exists(path) and overwrite is False:
            train_datalist, val_datalist = pickle.load(open(path, "rb"))
        else:
            data_dir = osp.join(data_root, paths["2D T1 npy subdir"])
            datalist = produce_datalist(data_dir, attrs, extra_kws=["date"])
            train_datalist, val_datalist = split_datalist_subj(datalist, train_fraction=.8007, slices=15)
            assert train_datalist[-1]["ID"][:10] != val_datalist[0]["ID"][:10]
            normalizations = normalize_datalist(train_datalist)
            normalize_datalist(val_datalist, normalizations)
            np.save(osp.join(ANALYSIS_DIR, "adni_normalizations.npy"), normalizations)
            pickle.dump((train_datalist, val_datalist), open(path, "wb"))

    else:
        raise NotImplementedError

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



def split_datalist(datalist, train_fraction):
    datalist = np.asarray(datalist)
    N = len(datalist)
    indices = list(range(N))
    np.random.shuffle(indices)
    train, val = indices[:train_fraction*N], indices[train_fraction*N:]
    return datalist[train], datalist[val]

def split_datalist_subj(datalist, train_fraction, slices):
    datalist = np.asarray(datalist)
    N = len(datalist)
    if N % slices != 0:
        raise ValueError
    # indices = list(range(0,N,slices))
    # np.random.shuffle(indices)
    # train, val = indices[:int(train_fraction*N)], indices[int(train_fraction*N):]
    # train = np.concatenate([np.asarray(train, dtype=int)+ix for ix in range(slices)])
    # val = np.concatenate([np.asarray(val, dtype=int)+ix for ix in range(slices)])
    # return datalist[train], datalist[val]
    train_N = int(train_fraction * N // slices * slices)
    return datalist[:train_N], datalist[train_N:]

def produce_datalist(data_dir, attrs, extra_kws=()):
    datapoints = pickle.load(open(data_dir+"/dataset.bin", "rb"))
    datalist = []
    for dp in sorted(datapoints, key=lambda dp:dp["ID"]):
        dp_att = []
        for attr in attrs:
            if attr in dp["observations"]:
                dp_att.append(dp["observations"][attr])
            else:
                dp_att.append(np.nan)

        datalist.append({"image": data_dir+"/"+dp["ID"]+".npy", "ID": dp["ID"], "attributes": dp_att,
            **{kw:dp[kw] for kw in extra_kws}})

    return datalist

def produce_datalist_mrigenie(data_dir, attrs):
    datapoints = pickle.load(open(data_dir+"/dataset.bin", "rb"))
    train_datalist = []
    val_datalist = []
    for dp in datapoints:
        dp_att = []
        for attr in attrs:
            if attr in dp["observations"]:
                dp_att.append(dp["observations"][attr])
            else:
                dp_att.append(np.nan)

        new_dp = {"image": data_dir+"/"+dp["ID"]+".npy", "ID": dp["ID"], "attributes": dp_att}
        if dp["clinical site"] == "04":
            train_datalist.append(new_dp)
        else:
            val_datalist.append(new_dp)

    return train_datalist, val_datalist

def normalize_datalist(datalist, normalizations=None):
    if normalizations is None:
        normalizations = []
        for ix in range(len(datalist[0]["attributes"])):
            observations = [dp["attributes"][ix] for dp in datalist]
            s = np.nanstd(observations)
            if s == 0:
                raise ValueError("no valid observations")
            normalizations.append((np.nanmean(observations), s))

    for dp in datalist:
        dp["attributes"] = [(dp["attributes"][ix] - mean) / std for ix, (mean, std) in enumerate(normalizations)]

    return normalizations
