import chainer
import chainer.functions as F


class TripletNet(chainer.Chain):

    def __init__(self, model):
        super(TripletNet, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, anchor, positive, negative):
        self.anchor = self.model(anchor)
        self.positive = self.model(positive)
        self.negative = self.model(negative)
        self.loss = F.triplet(anchor=self.anchor, positive=self.positive, negative=self.negative)

        return self.loss

