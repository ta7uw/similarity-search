import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


class MultiscaleNetModel(chainer.Chain):
    def __init__(self, n_class):
        self._layers = {}
        self.n_class = n_class

        # shallow layer
        self._layers["conv2_1"] = L.Convolution2D(None, 96, 3, stride=4, pad=1, wscale=0.02*np.sqrt(3*3*3))
        self._layers["norm2_1"] = L.BatchNormalization(96)
        self._layers["conv3_1"] = L.Convolution2D(None, 96, 3, stride=4, pad=1, wscale=0.02*np.sqrt(3*3*3))
        self._layers["norm3_1"] = L.BatchNormalization(96)

        #
        self._layers["fc4_1"] = L.Linear(4096, 4096)
        self._layers["fc4_2"] = L.Linear(4096, self.n_class)
