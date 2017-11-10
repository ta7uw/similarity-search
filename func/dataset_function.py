import glob
import chainer
from itertools import chain


def dataset_label():
    # Image directory name
    IMG_DIR = "dataset"

    # Each item image directory name
    b_names = glob.glob("{}/*".format(IMG_DIR))

    # List of path to image files
    fnames = [glob.glob("{}/*.jpg".format(b)) for b in b_names]
    fnames = list(chain.from_iterable(fnames))


