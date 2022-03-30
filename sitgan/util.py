import os, re, torch, cv2, sys, pdb, yaml
osp = os.path
from glob import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import monai.transforms as mtr

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)

def glob2(*paths):
    pattern = osp.expanduser(osp.join(*paths))
    if "*" not in pattern:
        pattern = osp.join(pattern, "*")
    return glob(pattern)

def format_float(x, n_decimals):
    if x == 0:
        return "0"
    elif np.isnan(x):
        return "NaN"
    if hasattr(x, "__iter__"):
        np.set_printoptions(precision=n_decimals)
        return str(np.array(x)).strip("[]")
    else:
        if n_decimals == 0:
            return ('%d'%x)
        else:
            return ('{:.%df}'%n_decimals).format(x)

def latex_mean_std(X=None, mean=None, stdev=None, n_decimals=1, percent=False, behaviour_if_singleton=None):
    if X is not None and len(X) == 1:
        mean = X[0]
        if not percent:
            return (r'{0:.%df}'%n_decimals).format(mean)
        else:
            return (r'{0:.%df}\%%'%n_decimals).format(mean*100)

    if stdev is None:
        mean = np.nanmean(X)
        stdev = np.nanstd(X)
    if not percent:
        return (r'{0:.%df}\pm {1:.%df}'%(n_decimals, n_decimals)).format(mean, stdev)
    else:
        return (r'{0:.%df}\%%\pm {1:.%df}\%%'%(n_decimals, n_decimals)).format(mean*100, stdev*100)

def flatten_list(collection):
    new_list = []
    for element in collection:
        new_list += list(element)
    return new_list
    
def parse_int_or_list(x):
    # converts string to an int or list of ints
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]

def parse_float_or_list(x):
    # converts string to a float or list of floats
    if not isinstance(x, str):
        return x
    try:
        return float(x)
    except ValueError:
        return [float(s.strip()) for s in x.split(',')]

class MetricTracker:
    def __init__(self, name=None, intervals=None, function=None, weight=1.):
        self.name = name
        self.epoch_history = {"train":[], "val":[]}
        self.intervals = intervals
        self.function = function
        self.minibatch_values = {"train":[], "val":[]}
        self.weight = weight
    
    def __call__(self, *args, phase="val", **kwargs):
        loss = self.function(*args, **kwargs)
        self.update_with_minibatch(loss, phase=phase)
        # if np.isnan(loss.mean().item()):
        #     raise ValueError(f"{self.name} became NaN")
        return loss.mean() * self.weight

    def update_at_epoch_end(self, phase="val"):
        if len(self.minibatch_values[phase]) != 0:
            self.epoch_history[phase].append(self.epoch_average(phase))
            self.minibatch_values[phase] = []

    def update_with_minibatch(self, value, phase="val"):
        if isinstance(value, torch.Tensor):
            if torch.numel(value) == 1:
                self.minibatch_values[phase].append(value.item())
            else:
                self.minibatch_values[phase] += list(value.detach().cpu().numpy())
        #elif not isanumber(value):
        elif not np.isnan(value):
            self.minibatch_values[phase].append(value)

    def epoch_average(self, phase="val"):
        if len(self.minibatch_values[phase]) != 0:
            return np.nanmean(self.minibatch_values[phase])
        elif len(self.epoch_history[phase]) != 0:
            return self.epoch_history[phase][-1]
        return np.nan

    def max(self, phase="val"):
        try: return np.nanmax(self.epoch_history[phase])
        except: return np.nan
    def min(self, phase="val"):
        try: return np.nanmin(self.epoch_history[phase])
        except: return np.nan

    def get_moving_average(self, window, interval=None, phase="val"):
        if interval is None:
            interval = window
        ret = np.cumsum(self.minibatch_values[phase], dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ret = ret[window - 1:] / window
        return ret[::interval]

    def is_at_max(self, phase="val"):
        return self.max(phase) >= self.epoch_average(phase)
    def is_at_min(self, phase="val"):
        return self.min(phase) <= self.epoch_average(phase)

    def histogram(self, path=None, phase="val", epoch=None):
        if epoch is None:
            epoch = len(self.epoch_history[phase]) * self.intervals[phase]
        if len(self.minibatch_values[phase]) < 5:
            return
        plt.hist(self.minibatch_values[phase])
        plt.title(f"{self.name}_{phase}_{epoch}")
        if path is not None:
            plt.savefig(path)
            plt.clf()

    def lineplot(self, path=None):
        _,axis = plt.subplots()

        if "train" in self.intervals:
            dx = self.intervals["train"]
            values = self.epoch_history["train"]
            if len(values) < 3:
                return
            x_values = np.arange(0, dx*len(values), dx)
            sns.lineplot(x=x_values, y=values, ax=axis, label="train")
        
        if "val" in self.intervals:
            dx = self.intervals["val"]
            values = self.epoch_history["val"]
            if len(values) < 3:
                return
            x_values = np.arange(0, dx*len(values), dx)
            sns.lineplot(x=x_values, y=values, ax=axis, label="val")

        axis.set_ylabel(self.name)

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            return axis

def save_plots(trackers, root=None):
    os.makedirs(root, exist_ok=True)
    for tracker in trackers:
        path = osp.join(root, tracker.name+".png")
        tracker.lineplot(path=path)


def save_metric_histograms(trackers, epoch, root):
    os.makedirs(root, exist_ok=True)
    for tracker in trackers:
        for phase in ["train", "val"]:
            path = osp.join(root, f"{epoch}_{tracker.name}_{phase}.png")
            tracker.histogram(path=path, phase=phase, epoch=epoch)


def save_examples(epoch, root, *imgs, transforms=None):
    imgs = list(imgs)
    if isinstance(imgs[0], torch.Tensor):
        for ix in range(len(imgs)):
            imgs[ix] = imgs[ix].detach().cpu().squeeze(1).numpy()
        if transforms is not None:
            transforms = transforms.detach().cpu().numpy()

    os.makedirs(root, exist_ok=True)
    for ix in range(imgs[0].shape[0]):
        cat = np.concatenate([img[ix] for img in imgs], axis=1)
        cat = rescale_clip(cat)
        if transforms is None:
            pass
        elif transforms.shape[1] == 3:
            dx, field = transforms[ix,0], transforms[ix,1:]
            dx = rescale_noclip(dx)
            field_mag = rescale_noclip(np.linalg.norm(field, axis=0))
            cat = np.concatenate((cat, dx, field_mag), axis=1)
        elif transforms.shape[1] == 2:
            field_mag = rescale_noclip(np.linalg.norm(transforms[ix], axis=0))
            cat = np.concatenate((cat, field_mag), axis=1)
        elif transforms.shape[1] == 1:
            dx = rescale_noclip(transforms[ix,0])
            cat = np.concatenate((cat, dx), axis=1)
        else:
            raise NotImplementedError("need to handle transforms")
        plt.imsave(f"{root}/{epoch}_{ix}.png", cat, cmap="gray")
    plt.close("all")

def save_example_3d(iteration, root, img, gt_seg, pred_logit):
    if isinstance(img, torch.Tensor):
        img = img.cpu().squeeze().numpy()
        gt_seg = gt_seg.cpu().squeeze().numpy()
        pred_logit = pred_logit.detach().cpu().squeeze().numpy()

    rescale = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
    img = rescale(img)

    os.makedirs(root, exist_ok=True)

    x = gt_seg.sum(axis=(1,2)).argmax()
    y = gt_seg.sum(axis=(0,2)).argmax()
    z = gt_seg.sum(axis=(0,1)).argmax()
    imgs = (img[x], img[:,y], img[:,:,z])
    gt_segs = (gt_seg[x], gt_seg[:,y], gt_seg[:,:,z])
    pred_logits = (pred_logit[x], pred_logit[:,y], pred_logit[:,:,z])

    for ix in range(3):
        img = imgs[ix]
        gt_seg = gt_segs[ix].round()
        pred_seg = pred_logits[ix] > 0
        plt.imsave(f"{root}/{iteration}_{ix}_img.png", img, cmap="gray")
        plt.imsave(f"{root}/{iteration}_{ix}_gt.png", gt_seg)
        plt.imsave(f"{root}/{iteration}_{ix}_logit.png", pred_logits[ix], cmap="gray")
        plt.imsave(f"{root}/{iteration}_{ix}_pred.png", pred_seg)

        img = draw_segs_as_contours(img, gt_seg, pred_seg)
        plt.imsave(f"{root}/{iteration}_{ix}_overlay.png", img)
    plt.close("all")

def draw_segs_as_contours(img, seg1, seg2, colors=((255,0,0), (0,0,255))):
    img = np.stack([img, img, img], -1)
    contour1 = cv2.findContours(seg1.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour1 = cv2.drawContours(np.zeros_like(img), contour1, -1, colors[0], 1)
    contour2 = cv2.findContours(seg2.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour2 = cv2.drawContours(np.zeros_like(img), contour2, -1, colors[1], 1)
    img *= (contour1 == 0) * (contour2 == 0)
    img += contour1 + contour2
    return img

def save_example_slices(img, gt_seg, pred_seg, root):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()
        gt_seg = gt_seg.detach().cpu().squeeze().numpy()
        pred_seg = pred_seg.detach().cpu().squeeze().numpy()

    rescale = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
    # rescale = mtr.ScaleIntensity(minv=0, maxv=255, dtype=np.uint8)
    img = rescale(img)

    os.makedirs(root, exist_ok=True)

    for z in range(5,76,5):
        img_slice = img[:,:,z]
        gt_seg_slice = gt_seg[:,:,z].round()
        pred_seg_slice = pred_seg[:,:,z] > 0
        plt.imsave(f"{root}/gt_{z}.png", gt_seg_slice)
        plt.imsave(f"{root}/pred_{z}.png", pred_seg_slice)

        img_slice = draw_segs_as_contours(img_slice, gt_seg_slice, pred_seg_slice)
        plt.imsave(f"{root}/{z}.png", img_slice)
    plt.close("all")



def latex_mean_std(X=None, mean=None, stdev=None, n_decimals=1, percent=False, behaviour_if_singleton=None):
    if X is not None and len(X) == 1:
        mean = X[0]
        if not percent:
            return (r'{0:.%df}'%n_decimals).format(mean)
        else:
            return (r'{0:.%df}\%%'%n_decimals).format(mean*100)

    if stdev is None:
        mean = np.nanmean(X)
        stdev = np.nanstd(X)
    if not percent:
        return (r'{0:.%df}\pm {1:.%df}'%(n_decimals, n_decimals)).format(mean, stdev)
    else:
        return (r'{0:.%df}\%%\pm {1:.%df}\%%'%(n_decimals, n_decimals)).format(mean*100, stdev*100)

def to_latex_table(table, cols=None, precision=3):
    if cols is None:
        cols = table.columns
    if not hasattr(precision, "__iter__"):
        precision = [precision] * len(cols)

    col_vals = {}
    ix = 0
    for col in cols:
        if A("array is numeric")(table[col]):
            if A("is an array of integers")(table[col]):
                col_vals[col] = A("format integers as string")(table[col].values, precision=precision[ix])
            else:
                col_vals[col] = A("format floats as string")(table[col].fillna(0).values, precision=precision[ix])
                ix += 1
        else:
            col_vals[col] = table[col].values

    outputs = []
    for ix in range(len(col_vals[col])):
        outputs.append(" & ".join([col_vals[c][ix] for c in cols]) + r" \\")
        print(outputs[-1])
    return outputs
