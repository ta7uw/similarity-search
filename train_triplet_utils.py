import chainer
from multi_scale_net import MultiscaleNet
from chainer import training
from chainer.training import extensions
from func.triplet_dataset_function import triplet_dataset_label, transform, create_triplet
from func.compute_mean import compute_mean
from triplet_net import TripletNet


class TripletDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base_path, crop_size, mean):
        similar_items, labels, fnames = triplet_dataset_label(base_path)

        self.crop_size = crop_size
        self.mean = mean
        self.triplets = create_triplet(similar_items, labels, fnames)

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

    similar_items, _, _ = triplet_dataset_label(train_data)
    multinet = MultiscaleNet(n_class=len(similar_items), pretrained_model=pre_trainedmodel)
    mean = compute_mean(dataset_path=train_data, insize=multinet.insize).mean(axis=(1, 2))
    multinet.mean = mean

    # Wrapping 'MultiscaleNet' model in 'TripletNet' model
    tripletnet = TripletNet(multinet)

    # Create the triplet dataset
    dataset = TripletDataset(train_data, multinet.insize, multinet.mean)
    train, val = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.8))

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu)
        tripletnet.to_gpu()

    if loaderjob <= 0:
        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        val_iter = chainer.iterators.SerialIterator(val, batchsize, shuffle=False, repeat=False)

    else:
        train_iter = chainer.iterators.MultiprocessIterator(dataset=train,
                                                            batch_size=batchsize,
                                                            n_processes=loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(dataset=val,
                                                          batch_size=batchsize,
                                                          n_processes=loaderjob,
                                                          shuffle=False,
                                                          repeat=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(tripletnet)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, "epoch"), out)

    val_interval = (val_iteration, "iteration")
    log_interval = (log_iteration, "iteration")

    trainer.extend(extensions.Evaluator(val_iter, tripletnet, device=gpu), trigger=val_interval)

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/loss',
         'main/accuracy',
         'validation/main/loss',
         'validation/main/accuracy',
         'elapsed_time',
         'lr']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(
        ["main/loss",
         "validation/main/loss"],
        x_key="epoch", file_name="loss.png"
    ))
    trainer.extend(extensions.PlotReport(
        ["main/accuracy",
         "validation/main/accuracy"],
        x_key="epoch", file_name="accuracy.png"))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(tripletnet, "model_epoch_{.updater.epoch}"),
                   trigger=(epoch, "epoch"))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()
