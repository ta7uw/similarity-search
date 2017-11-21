import glob
import os
from itertools import chain
import numpy as np
import random
from func.resize import resize
from PIL import Image


def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image



def triplet_dataset_label(path):
    """
        :param path: Path to the image dataset
        :return: b_names: Directory name of each item
                 labels: A unique ID is given each directory name
                 fnames: List of path to image files
        """
    # Path to the image dataset
    IMG_DIR = path

    # Directory name of each category
    category_names = glob.glob("{}/*".format(IMG_DIR))

    # Directory name of each item
    similar_items = glob.glob("{}/*".format(category_name) for category_name in category_names)

    # List of path to image files
    fnames = [glob.glob("{}/*.jpg".format(similar_name)) for similar_name in similar_items]
    fnames = list(chain.from_iterable(fnames))

    # A unique ID is given each directory name
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    similar_items = [os.path.basename(similar_item) for similar_item in similar_items]
    labels = [similar_items.index(l) for l in labels]
    return category_names, similar_items, labels, fnames


def transform(image_file_path, mean, crop_size, random=True):

    root="."
    full_path = os.path.join(root, image_file_path)
    image = _read_image_as_array(full_path, dtype=np.float32)

    if image.ndim == 2:
        # image is greyscale
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)

    image = resize(image, crop_size)
    image = image - mean[:, None, None]
    image *= (1.0 / 255.0)  # Scale to [0,1]
    image = image.astype(np.float32)

    if random:
        if random.randint(0, 1):
            image = image[:, :, ::-1]

    return image


def create_triplet(dataset):

    category_names, similar_items, labels, fnames = dataset
    triplets = []
    triplet_batch = 10

    for fname, label in zip(fnames, labels):

        anchor = fname
        anchor_label = label

        # Choose positive from same directory
        positive_list = np.where(labels == anchor_label)[0]
        positive_list = np.random.permutation(len(positive_list))[:triplet_batch]
        positive_list = list(fnames[i] for i in positive_list)

        # Choose negative image one by one from similar items other than itself.
        negative_list = np.where(labels != anchor_label)[0]
        negative_list = np.random.permutation(len(negative_list))[:triplet_batch]
        negative_list = list(fnames[i] for i in negative_list)

        for positive, negative in zip(positive_list, negative_list):
            triplet = anchor, positive, negative
            triplets.append(triplet)

    return triplets


