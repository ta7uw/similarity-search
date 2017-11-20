import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.initializers import constant, uniform
from .googlenetbn import GoogleNetBN


class MultiscaleNet(chainer.Chain):

    def __init__(self, n_class, pretrained_model=None,
                 mean=None, initialw=None, initialbias=None, googlenetbn_trianedmodel=None):
        self.n_class = n_class
        self.mean = mean
        self.initialbias = initialbias
        self.googlenetbn_trainedmodel = googlenetbn_trianedmodel

        self.insize = 224

        if mean is None:
            # imagenet means
            self.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]

        if initialw is None:
            # employ default initializers used in BVLC. For more detail, see
            self.initialW = uniform.LeCunUniform(scale=1.0)

        if pretrained_model:
            # As a sampling process is time-consuming
            # we employ a zero initializer for faster computation
            self.initialW = constant.Zero()

        super(MultiscaleNet, self).__init__()
        with self.init_scope():
            # Deep layers: GoogleNet of BatchNormalization version
            self.googlenetbn = GoogleNetBN(n_class=n_class)

            # Shallow layers
            self.conv_s1 = L.Convolution2D(None, 96, 3, stride=4, pad=1, initialW=0.02*np.sqrt(3*3*3))
            self.norm_s1 = L.BatchNormalization(96)
            self.conv_s2 = L.Convolution2D(None, 96, 3, stride=4, pad=1, initialW=0.02*np.sqrt(3*3*3))
            self.norm_s2 = L.BatchNormalization(96)

            # Final layers
            self.fc4_1 = L.Linear(None, 4096)
            self.fc4_2 = L.Linear(None, self.n_class)


    def __call__(self, x, t):
        # Deep layers
        h1 = F.max_pooling_2d(F.relu(self.googlenetbn.norm1(self.googlenetbn.conv1(x))), 3, stride=2, pad=1)
        h1 = F.max_pooling_2d(F.relu(self.googlenetbn.norm2(self.googlenetbn.conv2(h1))), 3, stride=2, pad=1)

        h1 = self.googlenetbn.inc3a(h1)
        h1 = self.googlenetbn.inc3b(h1)
        h1 = self.googlenetbn.inc3c(h1)
        h1 = self.googlenetbn.inc4a(h1)

        h1 = self.googlenetbn.inc4b(h1)
        h1 = self.googlenetbn.inc4c(h1)
        h1 = self.googlenetbn.inc4d(h1)

        h1 = self.googlenetbn.inc4e(h1)
        h1 = self.googlenetbn.inc5a(h1)
        h1 = F.average_pooling_2d(self.googlenetbn.inc5b(h1), 7)
        h1 = self.googlenetbn.loss3_fc(h1)

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




