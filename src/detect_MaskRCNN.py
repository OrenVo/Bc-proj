
import os

import sys
import random
import math
import json
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv


from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import log
from mrcnn.utils import Dataset

MODEL_DIR = os.path.abspath("logs")

class McConfig(Config):
    # Give the configuration a recognizable name
    NAME = "Mc"

    # Batch size = (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 bone

    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 1600

    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 6

    STEPS_PER_EPOCH = 200

    VALIDATION_STEPS = 15


class InterfaceMc(McConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

MODEL_DIR = 'data'

config = InterfaceMc()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
path = '../data/mask_rcnn_mc.h5'
model.load_weights(path, by_name=True)




def get_files(dir: str):
    """Creates and returns list of files in directory dir."""
    files = []
    for path, dirs, fs in os.walk(dir):
        files += [os.path.join(path, f) for f in fs if '.tif' in f]
    return files


def load_img(img_path):
    if not os.path.isfile(img_path):
        raise ValueError(f'path: \'{img_path}\' is not valid file path.')
    else:
        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f'Error while reading image \'{img_path}\'')
        else:
            return img
    return None


def get_mask(model, img):
    masks = model.detect([img], verbose=1)
    if masks:
        mask = masks[0]['masks']
        return mask
    return None


def create_binary_img(mask, img):
    result = np.zeros(img.shape)
    result[:] = 255
    # mask = (np.sum(mask, -1, keepdims=True) >= 1)
    r = np.reshape(mask, img.shape[:2])
    result[r] = np.array([0, 0, 0], dtype=np.uint8)
    return result

# %%

OUT_DIR = '../output/masks'
files = get_files('../data/dataset')
for f in files:
    img = load_img(f)
    mask = get_mask(model, img)
    out_f_path = os.path.join(OUT_DIR, f.split('/')[-1].split('.')[0] + '.npy')
    np.save(out_f_path, mask)


