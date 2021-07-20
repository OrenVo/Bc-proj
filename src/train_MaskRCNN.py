import cv2 as cv
import sys
import random
import math
import json
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath('Mask_RCNN'))
import imgaug
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import log
from mrcnn.utils import Dataset

ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "../data")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../data/mask_rcnn_coco.h5")

class McConfig(Config):
    # Give the configuration a recognizable name
    NAME = "Mc"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 bone

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 1600

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    #Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 6

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 15



class McDataset(utils.Dataset):
    """Loads metacarpal bones dataset and anotated data.
       """

    def load_mc(self, anot_dir):
        # Add classes and images to dataset
        self.add_class("Mc", 1, "mc bone")
        if not os.path.isdir(anot_dir):
            raise ValueError(f'{image_path} is not a valit file.')
        anot_files = []
        for path, dirs, files in os.walk(anot_dir):
            anot_files += [os.path.join(path, f) for f in files]
        # print(len(anot_files))
        for f in anot_files:
            with open(f) as handle:
                file = json.load(handle)
            polygons = []
            for shape in file['shapes']:
                if shape['label'] == 'mc':
                    polygons = [[int(x), int(y)] for x, y in shape['points']]
            image_path = os.path.join(anot_dir + "/../", file['imagePath'].split('../')[-1])
            if not os.path.isfile(image_path):
                raise ValueError(f'{image_path} is not a valit file.')
            height, width = file['imageHeight'], file['imageWidth']
            self.add_image(
                "Mc",
                image_id=image_path.split('/')[-1],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], 3], dtype=np.uint8)
        cv.fillPoly(mask, [np.array(info["polygons"], dtype=np.int32)], 1)
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


# image augmentations
augmentations = imgaug.augmenters.Sometimes(1 / 3, imgaug.augmenters.OneOf(
    [
        imgaug.augmenters.AdditiveGaussianNoise(scale=(0, 0.08 * 255)),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0)),
        imgaug.augmenters.GammaContrast((0.5, 2.0)),
        imgaug.augmenters.LogContrast(gain=(0.6, 1.4)),
        imgaug.augmenters.PiecewiseAffine(scale=(0.01, 0.05)),
        imgaug.augmenters.PerspectiveTransform(scale=(0.01, 0.15)),
        imgaug.augmenters.AverageBlur(k=(2, 8))
    ]
)
                                            )


def train(model, dataset='../data/train_data/annotation/', n_epochs=200):
    dataset_train = McDataset()
    dataset_train.load_mc(dataset + '/train_data/annotation')
    dataset_train.prepare()
    dataset_val = McDataset()
    dataset_val.load_mc(dataset + '/val_data/annotation')
    dataset_val.prepare()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=n_epochs,
                layers='heads',
                augmentation=augmentations
                )


if __name__ == '__main__':
    config = McConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    train(model)