import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.initializers import constant, uniform


class MultiscaleNet(chainer.Chain):

    def __init__(self, n_class, pretrained_model=None, mean=None, initialW=None, initialBias=None):
        self.n_class = n_class
        self.mean = mean
        self.initialbias = initialBias

        insize = 224

        if mean is None:
            # imagenet means
            self.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]

        if initialW is None:
            # employ default initializers used in BVLC. For more detail, see
            self.initialW = uniform.LeCunUniform(scale=1.0)

        if pretrained_model:
            # As a sampling process is time-consuming
            # we employ a zero initializer for faster computation
            self.initialW = constant.Zero()

        super(MultiscaleNet, self).__init__()
        with self.init_scope():
            # Deep layers: GoogleNet of BatchNormalization version
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3, initialW=self.initialW)
            self.norm1 = L.BatchNormalization(64)
            self.conv2_reduce = L.Convolution2D(None, 64, 1)
            self.conv2 = L.Convolution2D(None, 192, 3, stride=1, pad=1, initialW=self.initialW)
            self.norm2 = L.BatchNormalization(192)
            self.inc3a = L.InceptionBN(None, 64, 64, 64, 64, 96, "avg", 32)
            self.inc3b = L.InceptionBN(None, 64, 64, 96, 64, 96, "avg", 64)
            self.inc3c = L.InceptionBN(None, 0, 128, 160, 64, 96, "max", stride=2)
            self.inc4a = L.InceptionBN(None, 224, 64, 96, 96, 128, "avg", 128)
            self.inc4b = L.InceptionBN(None, 192, 96, 128, 96, 128, "avg", 128)
            self.inc4c = L.InceptionBN(None, 128, 128, 160, 128, 160, "avg", 128)
            self.inc4d = L.InceptionBN(None, 64, 128, 192, 160, 192, "avg", 128)
            self.inc4e = L.InceptionBN(None, 0, 128, 192, 192, 256, "max", stride=2)
            self.inc5a = L.InceptionBN(None, 352, 192, 320, 160, 224, "avg", 128)
            self.inc5b = L.InceptionBN(None, 352, 192, 320, 192, 224, "max", 128)
            self.loss3_fc = L.Linear(None, self.n_class, initialW=self.initialW)

            self.loss1_conv = L.Convolution2D(None, 128, 1, initialW=self.initialW)
            self.norma = L.BatchNormalization(128)
            self.loss1_fc1 = L.Linear(None, 1024, initialW=self.initialW)
            self.norma2 = L.BatchNormalization(1024)
            self.loss1_fc2 = L.Linear(None, self.n_class, initialW=self.initialW)

            self.loss2_conv = L.Convolution2D(None, 128, 1, initialW=self.initialW)
            self.normb = L.BatchNormalization(128)
            self.loss2_fc1 = L.Linear(None, 1024, initialW=self.initialW)
            self.normb2  = L.BatchNormalization(1024)
            self.loss2_fc2  = L.Linear(None, 1000, initialW=self.initialW)

            # Shallow layers
            self.conv_s1 = L.Convolution2D(None, 96, 3, stride=4, pad=1, initialW=0.02*np.sqrt(3*3*3))
            self.norm_s1 = L.BatchNormalization(96)
            self.conv_s2 = L.Convolution2D(None, 96, 3, stride=4, pad=1, initialW=0.02*np.sqrt(3*3*3))
            self.norm_s2 = L.BatchNormalization(96)

            # Final layers
            self.fc4_1 = L.Linear(None, 4096)
            self.fc4_2 = L.Linear(None, self.n_class)

    def __call__(self, anchor, p, n):
        self.anchor = self.forward_one(anchor)
        self.positive = self.forward_one(p)
        self.negative = self.forward_one(n)
        self.loss = F.triplet(anchor=self.anchor,
                              positive=self.positive,
                              negative=self.negative,
                              margin=0.2,
                              reduce="mean")
        return self.loss

    def forward_one(self, x, t):
        # Deep layers
        h1 = F.max_pooling_2d(F.relu(self.norm1(self.conv1(x))), 3, stride=2, pad=1)
        h1 = F.relu(self.conv2_reduce(h1))
        h1 = F.max_pooling_2d(F.relu(self.norm2(self.conv2(h1))), 3, stride=2, pad=1)

        h1 = self.inc3a(h1)
        h1 = self.inc3b(h1)
        h1 = self.inc3c(h1)
        h1 = self.inc4a(h1)

        a = F.average_pooling_2d(h1, 5, stride=3)
        a = F.relu(self.norma(self.loss1_conv(a)))
        a = F.relu(self.norma2(self.loss1_fc1(a)))
        a = self.loss1_fc2(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h1 = self.inc4b(h1)
        h1 = self.inc4c(h1)
        h1 = self.inc4d(h1)

        b = F.average_pooling_2d(h1, 5, stride=3)
        b = F.relu(self.normb(self.loss2_conv(b)))
        b = F.relu(self.normb2(self.loss2_fc1(b)))
        b = self.loss2_fc2(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h1 = self.inc4e(h1)
        h1 = self.inc5a(h1)
        h1 = F.average_pooling_2d(self.inc4e(h1), 7)
        h1 = self.loss3_fc(h1)
        loss3 = F.softmax_cross_entropy(h1, t)

        deep_loss = 0.03 * (loss1 + loss2) + loss3

        h1 = F.normalize(deep_loss)

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

        loss = F.softmax_cross_entropy(h, t)

        accuracy = F.accuracy(h, t)

        chainer.report({
            "deep_loss": deep_loss,
            "loss": loss,
            "accuracy": accuracy,
        }, self)
        return loss




