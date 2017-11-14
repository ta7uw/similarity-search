from tqdm import tqdm_notebook
import numpy as np
import chainer
import os
from .resize import resize
from .dataset_function import dataset_label


def compute_mean(dataset_path, insize):
    """
    Calculate mean image from dataset for training
    :param dataset_path: path to dataset for training
    :param insize: insize of model
    :return: value of mean iamge as array
    """
    _, labels, fnames = dataset_label(dataset_path)
    dataset = chainer.datasets.LabeledImageDataset(list(zip(fnames, labels)))
    if not os.path.exists("image_mean.npy"):
        # Calculate the mean with dataset that is not transformed version
        t, _ = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.8))

        mean = np.zeros((3, insize, insize))
        for img, _ in tqdm_notebook(t, desc="Calc man"):
            img = resize(img[:3].astype(np.uint8), insize)
            mean += img

        mean = mean / float(len(dataset))
        np.save("image_mean", mean)
    else:
        mean = np.load("image_mean.npy")

    return mean
