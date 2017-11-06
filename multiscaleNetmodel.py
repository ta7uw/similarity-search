import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.initializers import constant, uniform




class MultiscaleNetModel(chainer.Chain):

    def __init__(self, n_class):
        self._layers = {}
        self.n_class = n_class

        # employ default initializers used in BVLC. For more detail, see
        # https://github.com/chainer/chainer/pull/2424#discussion_r109642209
        kwargs = {'initialW': uniform.LeCunUniform(scale=1.0)}

        # Deep layers
        self.conv1 = L.Convolution2D(3, 64, 7, stride=2, pad=3, **kwargs),
        self.conv2_reduce = L.Convolution2D(64, 64, 1, **kwargs),
        self.conv2 = L.Convolution2D(64, 192, 3, stride=1, pad=1, **kwargs),
        self.inc3a = L.Inception(192, 64, 96, 128, 16, 32, 32),
        self.inc3b = L.Inception(256, 128, 128, 192, 32, 96, 64),
        self.inc4a = L.Inception(480, 192, 96, 208, 16, 48, 64),
        self.inc4b = L.Inception(512, 160, 112, 224, 24, 64, 64),
        self.inc4c = L.Inception(512, 128, 128, 256, 24, 64, 64),
        self.inc4d = L.Inception(512, 112, 144, 288, 32, 64, 64),
        self.inc4e = L.Inception(528, 256, 160, 320, 32, 128, 128),
        self.inc5a = L.Inception(832, 256, 160, 320, 32, 128, 128),
        self.inc5b = L.Inception(832, 384, 192, 384, 48, 128, 128),
        self.loss3_fc = L.Linear(1024, 1000, **kwargs),

        self.loss1_conv = L.Convolution2D(512, 128, 1, **kwargs),
        self.loss1_fc1 = L.Linear(2048, 1024, **kwargs),
        self.loss1_fc2 = L.Linear(1024, 1000, **kwargs),

        self.loss2_conv = L.Convolution2D(528, 128, 1, **kwargs),
        self.loss2_fc1 = L.Linear(2048, 1024, **kwargs),
        self.loss2_fc2 = L.Linear(1024, 1000, **kwargs)


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
        h1 = x
        h1 = F.normalize(h1)

        # Shallow layers
        h2 = F.average_pooling_2d(x, 4, stride=4, pad=2)
        h2 = F.max_pooling_2d(F.relu(self.norm_s1(self.conv_s1(h2))), 5, stride=4, pad=1)
        h3 = F.average_pooling_2d(x, 8, stride=8, pad=4)
        h3 = F.max_pooling_2d(F.relu(self.norm_s2(self.conv_s2(h3))), 4, stride=2, pad=1)

        h23 = F.concat((h2, h3), axis=1)
        h23 = F.normalize(F.reshape(h23, (x.data.shape[0], 3072)))

        h = F.concat((h1, h23), axis=1)

        h = F.normalize(F.relu(self.fc4_1(h)))
        h = self.fc4_2(h)
        return h




