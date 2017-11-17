import glob
import os
from itertools import chain
import numpy as np
import random
from func.resize import resize
from chainer.datasets import ImageDataset


def triplet_dataset_label(path):
    """
        :param path: Path to the image dataset
        :return: b_names: Directory name of each item
                 labels: A unique ID is given each directory name
                 fnames: List of path to image files
        """
    # Path to the image dataset
    IMG_DIR = path

    # Directory name of each item
    category_names = glob.glob("{}/*".format(IMG_DIR))

    # Directory name of each item
    similar_items = glob.glob("{}/*".format(category_names))

    # List of path to image files
    fnames = [glob.glob("{}/*.jpg".format(similar_name)) for similar_name in similar_items]
    fnames = list(chain.from_iterable(fnames))

    # A unique ID is given each directory name
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    b_names = [os.path.basename(similar_name) for similar_name in similar_items]
    labels = [b_names.index(l) for l in labels]
    return b_names, labels, fnames


def transform(image_file_path, mean, crop_size, random=True):
    image = ImageDataset(image_file_path)
    image = resize(image, crop_size)
    image = image - mean[:, None, None]
    image *= (1.0 / 255.0)  # Scale to [0,1]
    image = image.astype(np.float32)

    if random:
        if random.randint(0, 1):
            image = image[:, :, ::-1]

    return image

