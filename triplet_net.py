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
        :param anchor:
        :param positive:
        :param negative:
        :return:
            ~chainer.varibales:
                A variable holding a scalar that is the loss value calculated.
        """
        self.anchor = self.model(anchor)
        self.positive = self.model(positive)
        self.negative = self.model(negative)
        # Using margin = 0.2, reduce = "mean" to triplet function
        self.loss = F.triplet(anchor=self.anchor, positive=self.positive, negative=self.negative)

        return self.loss

