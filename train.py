import argparse
from train_googlenetbn_utils import train_run
from train_triplet_utils import train_triplet


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",default=0, type=int,
                        help="Please 0 or 1."
                             "If you select 0, train googlnet model."
                             "If you select 1, train tripletnet")

    parser.add_argument("--train", default="dataset",
                        help="Path to the root directory of the training dataset")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="The size of batch")

    parser.add_argument("--epoch", type=int, default=20,
                        help="Number of epochs to train")

    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")

    parser.add_argument("--out", default="result",
                        help="The directory in which logs are saved")

    parser.add_argument("--val_iteration", type=int, default=1000,
                        help="The number of iterations between every validation")

    parser.add_argument("--loaderjob", type=int, default=4,
                        help="The number of processes to launch for MultiprocessIterator")

    parser.add_argument("--log_iteration", type=int, default=100,
                        help="The number of iterations between every logging")

    parser.add_argument("--resume",
                        help="The path to the trainer snapshot to resume from."
                             "If unseprcified, no shapshot will be resumed")

    args = parser.parse_args()

    print("Training strats")
    if args.model == 0:
        train_run(
            train_data=args.train,
            epoch=args.epoch,  batchsize=args.batch_size,
            gpu=args.gpu, out=args.out, val_iteration=args.val_iteration,
            log_iteration=args.log_iteration, loaderjob=args.loaderjob,
            resume=args.resume, pre_trainedmodel=True
        )

    elif args.model == 1:
        # Set trained model param for GoogleNet BatchNormalization version
        googlenetbn_trainedmodel = "model_epoch_20"
        train_triplet(
            train_data=args.train,
            epoch=args.epoch, batchsize=args.batch_size,
            gpu=args.gpu, out=args.out, val_iteration=args.val_iteration,
            log_iteration=args.log_iteration, loaderjob=args.loaderjob,
            resume=args.resume, pre_trainedmodel=True, googlenetbn_trainedmodel=googlenetbn_trainedmodel
        )


if __name__ == '__main__':
    main()