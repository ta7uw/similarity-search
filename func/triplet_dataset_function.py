import glob
import os
from itertools import chain


def main(path):
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
    similar_names = glob.glob("{}/{}/*".format(IMG_DIR, category_names))

    # List of path to image files
    fnames = [glob.glob("{}/*.jpg".format(similar_name)) for similar_name in similar_names]
    fnames = list(chain.from_iterable(fnames))

    # A unique ID is given each directory name
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    b_names = [os.path.basename(similar_name) for similar_name in similar_names]
    labels = [b_names.index(l) for l in labels]
    return b_names, labels, fnames
