import argparse
import numpy as np

import chainer
from train_utils import train, PreprocessedDataset
from googlenetbn import GoogleNetBN



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        help="Path to the root directory of the training dataset")

    parser.add_argument("--val",
                        help="Path to the root directory of the validation dataset. If this is not supplied,"
                             "the data for train dataset is split into two wtih ratio 8:2. ")

    parser.add_argument("--iteration", type=int, default=120000,
                        help="the number of iterations to run until finishing the train loop")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")

    parser.add_argument("--step_size", type=int, default=-1,
                        help="The number of iterations to run before dropping the learning rate by 0.1")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="The size of batch")

    parser.add_argument("--epoch", type=int, default=20,
                        help="Number of epochs to train")

    parser.add_argument("--mean", default="mean.npy",
                        help="Mean file (computed by compute_mean.py)")

    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU ID")

    parser.add_argument("--out", default="result",
                        help="The directory in which logs are saved")

    parser.add_argument("--val_iteration", type=int, default=10000,
                        help="The number of iterations between every validation")

    parser.add_argument("--loaderjob", type=int, default=4,
                        help="The number of processes to launch for MultiprocessIterator")

    parser.add_argument("--log_iteration", type=int, default=100,
                        help="The number of iterations between every logging")

    parser.add_argument("--resume",
                        help="The path to the trainer snapshot to resume from."
                             "If unseprcified, no shapshot will be resumed")

    args = parser.parse_args()
    mean = np.load(args.mean)

    model = GoogleNetBN()

    if args.val is not None:
        train_dataset = PreprocessedDataset(args.train, args.root, mean, model.insize)
        val_dataset = PreprocessedDataset(args.val, args.root, mean, model.insize, False)
    else:
        # If --val is not supplied, the train data is split in two with ratio 8:2
        dataset = args.train
        train_data, val_data = chainer.datasets.split_dataset_random(dataset, int(len(dataset)*0.8))
        train_dataset = PreprocessedDataset(train_data, args.root, mean, model.insize)
        val_dataset = PreprocessedDataset(val_data, args.root, mean, model.insize, False)

    print("Training strats")
    train(
        train_data=train_dataset, val_data=val_dataset,
        iteration=args.iteration, epoch=args.epoch, lr=args.lr,
        step_size=args.stepsoze, batchsize=args.batchsize,
        gpu=args.gpu, out=args.out, val_iteration=args.val_iteration,
        log_iteration=args.log_iteration, loaderjob=args.loaderjob,
        resume=args.resume, pre_trainedmodel=True
    )


if __name__ == '__main__':
    main()