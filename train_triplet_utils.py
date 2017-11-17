import chainer
from googlenetbn import GoogleNetBN
from multi_scale_net import MultiscaleNet
from chainer import training
from chainer.training import extensions


class TripletDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base_path, file_name, triplet_file_name, transform, laoder):
        self.base_path = base_path
        self.filenamelist = []



    def __len__(self):
        return

    def get_example(self, i):




def train_triplet(train_data, epoch, batchsize,
               gpu, out, val_iteration, log_iteration, loaderjob,
              resume, pre_trainedmodel=True):


    MultiNet = MultiscaleNet()
