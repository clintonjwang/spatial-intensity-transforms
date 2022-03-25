import os
import shutil
import re
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import util

IMG_DIR = 'img'
LABEL_DIR = 'label'

def preprocess(data_paths, shape=(112, 112, 80), distr=(30, 10), labels=False):
    """
    Preprocesses the data by resizing (crops out of frame and pads images that are too small)
    as well as adding noise to any introduced padding. 

    Params:
    data_paths: a list of directories to process
    shape: the shape to resize to
    distr: the distribution of the values to pad the images with
    labels: bool that is True if data_paths contains label directories, False otherwise. 
    TODO: currently the data_paths list must contain either all label directories or all
    image directories, it cannot handle both. 
    """
    for dir in data_paths:
        processed = '{}_processed'.format(dir)
        if not os.path.exists(processed):
            os.mkdir(processed)
    for dir in data_paths:
        for f in os.listdir(dir):
            data = util.load_img(os.path.join(dir, f))
            resized = util.crop_or_pad(data, shape, distr, labels=labels)
            util.save_img(resized, '{}_processed'.format(dir), f)

if __name__ == "__main__":
    data_paths = ['/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/new/img']
    preprocess(data_paths)
    data_paths = ['/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/new/label']
    preprocess(data_paths,labels=True)
