import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.initializers import constant, uniform


class MultiscaleNetModel(chainer.Chain):

    def __init__(self, n_class):
        self._layers = {}
        self.n_class = n_class

        insize = 224

        # employ default initializers used in BVLC. For more detail, see
        # https://github.com/chainer/chainer/pull/2424#discussion_r109642209
        self.initialW = uniform.LeCunUniform(scale=1.0)

        # Deep layers: GoogleNet of BatchNormalization version
        self._layers["conv1"] = L.Convolution2D(None, 64, 7, stride=2, pad=3,initialW=self.initialW)
        self._layers["norm1"] = L.BatchNormalization(64)
        self._layers["conv2_reduce"] = L.Convolution2D(None, 64, 1, initialW=self.initialW)
        self._layers["conv2"] = L.Convolution2D(None, 192, 3, stride=1, pad=1, initialW=self.initialW)
        self._layers["norm2"] = L.BatchNormalization(192)
        self._layers["inc3a"] = L.Inception(None, 64, 96, 128, 16, 32, 32)
        self._layers["inc3b"] = L.Inception(None, 128, 128, 192, 32, 96, 64)
        self._layers["inc4a"] = L.Inception(None, 192, 96, 208, 16, 48, 64)
        self._layers["inc4b"] = L.Inception(None, 128, 128, 256, 24, 64, 64)
        self._layers["inc4d"] = L.Inception(None, 112, 144, 288, 32, 64, 64)
        self._layers["inc4e"] = L.Inception(None, 256, 160, 320, 32, 128, 128)
        self._layers["inc5a"] = L.Inception(None, 256, 160, 320, 32, 128, 128)
        self._layers["inc5b"] = L.Inception(None, 384, 192, 384, 48, 128, 128)
        self._layers["loss3_fc"] = L.Linear(None, self.n_class, initialW=self.initialW)

        self._layers["loss1_conv"] = L.Convolution2D(None, 128, 1, initialW=self.initialW)
        self._layers["norma"] = L.BatchNormalization(128)
        self._layers["loss1_fc1"] = L.Linear(None, 1024, initialW=self.initialW)
        self._layers["norma2"] = L.BatchNormalization(1024)
        self._layers["loss1_fc2"] = L.Linear(None, self.n_class, initialW=self.initialW)

        self._layers["loss2_conv"] = L.Convolution2D(None, 128, 1, initialW=self.initialW)
        self._layers["normb"] = L.BatchNormalization(128)
        self._layers["loss2_fc1"] = L.Linear(None, 1024, initialW=self.initialW)
        self._layers["normb2"] = L.BatchNormalization(1024)
        self._layers["loss2_fc2"] = L.Linear(None, 1000, initialW=self.initialW)

        # Shallow layers
        self._layers["conv2_1"] = L.Convolution2D(None, 96, 3, stride=4, pad=1, initialW=0.02*np.sqrt(3*3*3))
        self._layers["norm2_1"] = L.BatchNormalization(96)
        self._layers["conv3_1"] = L.Convolution2D(None, 96, 3, stride=4, pad=1, initialW=0.02*np.sqrt(3*3*3))
        self._layers["norm3_1"] = L.BatchNormalization(96)

        # Final layers
        self._layers["fc4_1"] = L.Linear(4096, 4096)
        self._layers["fc4_2"] = L.Linear(4096, self.n_class)

        super(MultiscaleNetModel, self).__init__(**self._layers)

    def __call__(self, x):

        # Deep layers
        h1 = F.max_pooling_2d(F.relu(self._layers["norm1"](self._layers["conv1"](x))), 3, stride=2, pad=1)

        h1 = F.normalize(h1)

        # Shallow layers
        h2 = F.average_pooling_2d(x, 4, stride=4, pad=2)
        h2 = F.max_pooling_2d(F.relu(self.norm2_1(self.conv_s1(h2))), 5, stride=4, pad=1)
        h3 = F.average_pooling_2d(x, 8, stride=8, pad=4)
        h3 = F.max_pooling_2d(F.relu(self.norm3_1(self.conv_s2(h3))), 4, stride=2, pad=1)

        h23 = F.concat((h2, h3), axis=1)
        h23 = F.normalize(F.reshape(h23, (x.data.shape[0], 3072)))

        h = F.concat((h1, h23), axis=1)

        h = F.normalize(F.relu(self.fc4_1(h)))
        h = self.fc4_2(h)
        return h




