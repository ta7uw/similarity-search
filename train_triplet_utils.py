import chainer
from googlenetbn import GoogleNetBN
from multi_scale_net import MultiscaleNet
from chainer import training
from chainer.training import extensions
from func.triplet_dataset_function import triplet_dataset_label, transform
from func.resize import resize


class TripletDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base_path, crop_size, mean):
        _, labels, fnames = triplet_dataset_label(base_path)
        self.triplets = []
        self.crop_size = crop_size
        self.mean = mean

    def __len__(self):
        return len(self.triplets)

    def get_example(self, i):
        path1, path2, path3 = self.triplets[i]
        anchor = transform(path1, self.mean, self.crop_size)
        positive = transform(path1, self.mean, self.crop_size)
        negative = transform(path1, self.mean, self.crop_size)
        return anchor, positive, negative


def train_triplet(train_data, epoch, batchsize,
               gpu, out, val_iteration, log_iteration, loaderjob,
              resume, pre_trainedmodel=True):


    MultiNet = MultiscaleNet()
