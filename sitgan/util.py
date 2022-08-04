import os, torch, sys, yaml
osp = os.path
import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import monai.transforms as mtr

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)

def get_num_channels_for_outputs(outputs):
    if "displacement" in outputs or "velocity" in outputs:
        if "," in outputs:
            out_channels = 3
        else:
            out_channels = 2
    else:
        out_channels = 1
    return out_channels

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
