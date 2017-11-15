import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer.initializers import constant, uniform


class GoogleNetBN(chainer.Chain):
    """
    GoogleNet of BatchNormalization version
    """

    def __init__(self, n_class=None, pretrained_model=None, mean=None, initialW=None, initialBias=None):
        self.n_class = n_class
        self.mean = mean
        self.initialbias = initialBias

        self.insize = 224

        if n_class is None:
            self.n_class = 100

        if mean is None:
            # imagenet means
            self.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]

        if initialW is None:
            # employ default initializers used in BVLC. For more detail, see
            self.initialW = uniform.LeCunUniform(scale=1.0)

        if pretrained_model is None:
            # As a sampling process is time-consuming
            # we employ a zero initializer for faster computation
            self.initialW = constant.Zero()

        super(GoogleNetBN, self).__init__()
        with self.init_scope():
            # Deep layers: GoogleNet of BatchNormalization version
            self.conv1 = L.Convolution2D(None, 64, 7, stride=2, pad=3, nobias=True)
            self.norm1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(None, 192, 3, stride=1, pad=1, nobias=True)
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

            self.loss1_conv = L.Convolution2D(None, 128, 1, initialW=self.initialW, nobias=True)
            self.norma = L.BatchNormalization(128)
            self.loss1_fc1 = L.Linear(None, 1024, initialW=self.initialW, nobias=True)
            self.norma2 = L.BatchNormalization(1024)
            self.loss1_fc2 = L.Linear(None, self.n_class, initialW=self.initialW)

            self.loss2_conv = L.Convolution2D(None, 128, 1, initialW=self.initialW, nobias=True)
            self.normb = L.BatchNormalization(128)
            self.loss2_fc1 = L.Linear(None, 1024, initialW=self.initialW, nobias=True)
            self.normb2 = L.BatchNormalization(1024)
            self.loss2_fc2 = L.Linear(None, self.n_class, initialW=self.initialW)

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.norm1(self.conv1(x))), 3, stride=2, pad=1)
        h = F.max_pooling_2d(F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.loss1_conv(a)))
        a = F.relu(self.norma2(self.loss1_fc1(a)))
        a = self.loss1_fc2(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.loss2_conv(b)))
        b = F.relu(self.normb2(self.loss2_fc1(b)))
        b = self.loss2_fc2(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.loss3_fc(h)
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.03 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            "loss": loss,
            "loss1": loss1,
            "loss2": loss2,
            "loss3": loss3,
            "accuracy": accuracy,
        }, self)

        return loss

    def predict(self, x):
        h = F.max_pooling_2d(F.relu(self.norm1(self.conv1(x))), 3, stride=2, pad=1)
        h = F.max_pooling_2d(F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.loss3_fc(h)

        return F.softmax(h)
