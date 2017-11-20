import chainer
import chainer.functions as F


class TripletNet(chainer.Chain):
    """
    This is triplet networks
    """
    def __init__(self, model):
        super(TripletNet, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, anchor, positive, negative):
        """
        It takes a triplet of variables as inputs
        :param anchor: The anchor example variable. The shape should be (N, K),
                where N denotes the minibatch size, and K denotes the dimension of the anchor.
        :param positive: The positive example variable. The shape should be the same as anchor.
        :param negative: The negative example variable. The shape should be the same as anchor.
        :return:
            Type:~chainer.varibales:
            A variable holding a scalar that is the loss value calculated.
        """
        self.anchor = self.model(anchor)
        self.positive = self.model(positive)
        self.negative = self.model(negative)
        # Using margin = 0.2, reduce = "mean" to triplet function
        self.loss = F.triplet(anchor=self.anchor, positive=self.positive, negative=self.negative)

        return self.loss

    def project(self, x):
        return self.model(x)

