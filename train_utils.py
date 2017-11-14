import random

import chainer
from chainer import training
from chainer.training import extensions

from googlenetbn import GoogleNetBN
from func.dataset_function import dataset_label
from func.compute_mean import compute_mean
from func.resize import resize


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset_path, mean, crop_size, random=True):
        _, labels, fnames = dataset_label(dataset_path)
        self.base = chainer.datasets.LabeledImageDataset(list(zip(fnames, labels)))
        self.mean = mean
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size
        image, label = self.base[i]
        image = image.transpose(1, 2, 0)
        image = resize(image, crop_size)
        _, h, w = image.shape

        if self.random:
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]

        else:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 2555.0)  # Scale to [0, 1]
        return image, label


def train_run(train_data, epoch, batchsize,
               gpu, out, val_iteration, log_iteration, loaderjob,
              resume, pre_trainedmodel=True):

    b_names, labels, _ = dataset_label(train_data)
    model = GoogleNetBN(n_class=len(b_names))

    mean = compute_mean(dataset_path=train_data, insize=model.insize)

    dataset = PreprocessedDataset(train_data, mean,  model.insize)
    train, val = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.8))

    if pre_trainedmodel:
        print("Load model")
        chainer.serializers.load_npz("tuned_googlenetbn.npz", model)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    if loaderjob <= 0:
        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        val_iter = chainer.iterators.SerialIterator(val, batchsize, shuffle=False, repeat=False)

    else:
        train_iter = chainer.iterators.MultiprocessIterator(train,
                                                            batchsize,
                                                            n_processes=loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(val,
                                                          batchsize,
                                                          n_processes=loaderjob,
                                                          shuffle=False,
                                                          repeat=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, "epoch"), out)

    val_interval = (val_iteration, "iteration")
    log_interval = (log_iteration, "iteration")

    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu), trigger=val_interval)

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
    trainer.extend(extensions.snapshot_object(model, "model_iter_{.updater.iteration}"),
                   trigger=val_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()









