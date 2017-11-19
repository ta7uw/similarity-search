import chainer
from googlenetbn import GoogleNetBN
from multi_scale_net import MultiscaleNet
from chainer import training
from chainer.training import extensions
from func.triplet_dataset_function import triplet_dataset_label, transform, create_triplet
from func.resize import resize


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

    MultiNet = MultiscaleNet()

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
