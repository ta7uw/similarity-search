import numpy as np

import chainer
from chainer import training
from chainer.training import extensions, triggers


from model import MultiscaleNet


def train(train_data, val_data, iteration, lr, step_size, batchsize,
          gpu, out, val_iteration, log_iteration, loaderjob, resume):

    model = MultiscaleNet()

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    if loaderjob <= 0:
        train_iter = chainer.iterators.SerialIterator(train_data,
                                                      batchsize)
    else:
        train_iter = chainer.iterators.MultiprocessIterator(train_data,
                                                            batchsize,
                                                            n_processes=min(loaderjob, batchsize))

    val_iter = chainer.iterators.SerialIterator(val_data, batchsize, shuffle=False, repeat=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (iteration, "iteration"), out)

    trainer.extend(extensions.ExponentialShift("lr", 0.1, init=lr),
                   trigger=triggers.ManualScheduleTrigger(step_size, "iteration"))

    val_interval = (val_iteration, "iteration")
    log_interval = (log_iteration, "iteration")
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

    if resume:
        chainer.serializers.load_npz(resume, trainer)

    trainer.run()









