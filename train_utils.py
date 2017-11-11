import numpy as np
import random

import chainer
from chainer import training
from chainer.training import extensions, triggers

from googlenetbn import GoogleNetBN
from func.dataset_function import dataset_label


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype("f")
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return self.base

    def get_example(self, i):
        crop_size = self.crop_size

        image, label = self.base[i]
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
        image *= (1.0 // 2555.0)  # Scale to [0, 1]
        return image, label


def train(train_data, val_data, iteration, epoch,  lr, step_size, batchsize,
          gpu, out, val_iteration, log_iteration, loaderjob, resume, pre_trainedmodel = True):

    b_names, labels = dataset_label()
    model =GoogleNetBN(n_class=len(b_names))

    if pre_trainedmodel:
        print("Load model")
        chainer.serializers.load_npz("tuned_googlenetbn.npz", model)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    if loaderjob <= 0:
        train_iter = chainer.iterators.SerialIterator(train_data, batchsize)
        val_iter = chainer.iterators.SerialIterator(val_data, batchsize, shuffle=False, repeat=False)

    else:
        train_iter = chainer.iterators.MultiprocessIterator(train_data,
                                                            batchsize,
                                                            n_processes=loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(val_data,
                                                          batchsize,
                                                          n_processes=loaderjob,
                                                          shuffle=False,
                                                          repeat=False)

    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, "epoch"), out)

    trainer.extend(extensions.ExponentialShift("lr", 0.1, init=lr),
                   trigger=triggers.ManualScheduleTrigger(step_size, "iteration"))

    val_interval = (val_iteration, "iteration")
    log_interval = (log_iteration, "iteration")

    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu), trigger=val_interval)

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ["epoch", "itertion", "lr", "main/loss", "elapsed_time"]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(
        ["main/accuracy", "validation/main/accuracy"], x_key="epoch", file_name="accuracy.png"))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, "model_iter_{.updater.iteration}"),
                   trigger=val_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()









