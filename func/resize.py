import numpy as np
from PIL import Image


def resize(image, insize):
    image = Image.fromarray(image.transpose(1, 2, 0))
    image = image.resize((insize, insize), Image.BICUBIC)

    return np.asanyarray(image).transpose(2, 0, 1)