from chainer.links.caffe import CaffeFunction
from chainer import serializers
from googlenetbn import GoogleNetBN
from func.dataset_function import dataset_label
import argparse


def convert_caffe2chainer():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    print('start loading model file...')
    caffe_model = CaffeFunction('googlenet.caffemodel')
    print('Done.')

    # copy parameters from caffemodel into chainer model
    print('start copy params.')
    b, _, _ = dataset_label(args.dataset)
    googlenet = GoogleNetBN(n_class=len(b))

    googlenet.conv1.W.data = caffe_model['conv1/7x7_s2'].W.data
    googlenet.conv2.W.data = caffe_model['conv2/3x3'].W.data

    """Inception module of the new GoogLeNet with BatchNormalization."""
    # inc3a
    googlenet.inc3a.conv1.W.data = caffe_model['inception_3a/1x1'].W.data
    googlenet.inc3a.conv3.W.data = caffe_model['inception_3a/3x3'].W.data
    googlenet.inc3a.conv33a.W.data = caffe_model['inception_3a/double3x3a'].W.data
    googlenet.inc3a.conv33b.W.data = caffe_model['inception_3a/double3x3b'].W.data
    googlenet.inc3a.proj3.W.data = caffe_model['inception_3a/3x3_reduce'].W.data
    googlenet.inc3a.proj33.W.data = caffe_model['inception_3a/double3x3_reduce'].W.data
    googlenet.inc3a.poolp.W.data = caffe_model['inception_3a/pool_proj'].W.data

    # inc3b
    googlenet.inc3b.conv1.W.data = caffe_model['inception_3b/1x1'].W.data
    googlenet.inc3b.conv3.W.data = caffe_model['inception_3b/3x3'].W.data
    googlenet.inc3b.conv33a.W.data = caffe_model['inception_3b/double3x3a'].W.data
    googlenet.inc3b.conv33b.W.data = caffe_model['inception_3b/double3x3b'].W.data
    googlenet.inc3b.proj3.W.data = caffe_model['inception_3b/3x3_reduce'].W.data
    googlenet.inc3b.proj33.W.data = caffe_model['inception_3b/double3x3_reduce'].W.data
    googlenet.inc3b.poolp.W.data = caffe_model['inception_3b/pool_proj'].W.data

    # inc3c
    googlenet.inc3c.conv3.W.data = caffe_model['inception_3c/3x3'].W.data
    googlenet.inc3c.conv33a.W.data = caffe_model['inception_3c/double3x3a'].W.data
    googlenet.inc3c.conv33b.W.data = caffe_model['inception_3c/double3x3b'].W.data
    googlenet.inc3c.proj3.W.data = caffe_model['inception_3c/3x3_reduce'].W.data
    googlenet.inc3c.proj33.W.data = caffe_model['inception_3c/double3x3_reduce'].W.data

    # inc4a
    googlenet.inc4a.conv1.W.data = caffe_model['inception_4a/1x1'].W.data
    googlenet.inc4a.conv3.W.data = caffe_model['inception_4a/3x3'].W.data
    googlenet.inc4a.conv33a.W.data = caffe_model['inception_4a/double3x3a'].W.data
    googlenet.inc4a.conv33b.W.data = caffe_model['inception_4a/double3x3b'].W.data
    googlenet.inc4a.proj3.W.data = caffe_model['inception_4a/3x3_reduce'].W.data
    googlenet.inc4a.proj33.W.data = caffe_model['inception_4a/double3x3_reduce'].W.data
    googlenet.inc4a.poolp.W.data = caffe_model['inception_4a/pool_proj'].W.data


    # inc4b
    googlenet.inc4b.conv1.W.data = caffe_model['inception_4b/1x1'].W.data
    googlenet.inc4b.conv3.W.data = caffe_model['inception_4b/3x3'].W.data
    googlenet.inc4b.conv33a.W.data = caffe_model['inception_4b/double3x3a'].W.data
    googlenet.inc4b.conv33b.W.data = caffe_model['inception_4b/double3x3b'].W.data
    googlenet.inc4b.proj3.W.data = caffe_model['inception_4b/3x3_reduce'].W.data
    googlenet.inc4b.proj33.W.data = caffe_model['inception_4b/double3x3_reduce'].W.data
    googlenet.inc4b.poolp.W.data = caffe_model['inception_4b/pool_proj'].W.data

    # inc4c
    googlenet.inc4c.conv1.W.data = caffe_model['inception_4c/1x1'].W.data
    googlenet.inc4c.conv3.W.data = caffe_model['inception_4c/3x3'].W.data
    googlenet.inc4c.conv33a.W.data = caffe_model['inception_4c/double3x3a'].W.data
    googlenet.inc4c.conv33b.W.data = caffe_model['inception_4c/double3x3b'].W.data
    googlenet.inc4c.proj3.W.data = caffe_model['inception_4c/3x3_reduce'].W.data
    googlenet.inc4c.proj33.W.data = caffe_model['inception_4c/double3x3_reduce'].W.data
    googlenet.inc4c.poolp.W.data = caffe_model['inception_4c/pool_proj'].W.data

    # inc4d
    googlenet.inc4d.conv1.W.data = caffe_model['inception_4d/1x1'].W.data
    googlenet.inc4d.conv3.W.data = caffe_model['inception_4d/3x3'].W.data
    googlenet.inc4d.conv33a.W.data = caffe_model['inception_4d/double3x3a'].W.data
    googlenet.inc4d.conv33b.W.data = caffe_model['inception_4d/double3x3b'].W.data
    googlenet.inc4d.proj3.W.data = caffe_model['inception_4d/3x3_reduce'].W.data
    googlenet.inc4d.proj33.W.data = caffe_model['inception_4d/double3x3_reduce'].W.data
    googlenet.inc4d.poolp.W.data = caffe_model['inception_4d/pool_proj'].W.data

    # inc4e
    googlenet.inc4e.conv3.W.data = caffe_model['inception_4e/3x3'].W.data
    googlenet.inc4e.conv33a.W.data = caffe_model['inception_4e/double3x3a'].W.data
    googlenet.inc4e.conv33b.W.data = caffe_model['inception_4e/double3x3b'].W.data
    googlenet.inc4e.proj3.W.data = caffe_model['inception_4e/3x3_reduce'].W.data
    googlenet.inc4e.proj33.W.data = caffe_model['inception_4e/double3x3_reduce'].W.data

    # inc5a
    googlenet.inc5a.conv1.W.data = caffe_model['inception_5a/1x1'].W.data
    googlenet.inc5a.conv3.W.data = caffe_model['inception_5a/3x3'].W.data
    googlenet.inc5a.conv33a.W.data = caffe_model['inception_5a/double3x3a'].W.data
    googlenet.inc5a.conv33b.W.data = caffe_model['inception_5a/double3x3b'].W.data
    googlenet.inc5a.proj3.W.data = caffe_model['inception_5a/3x3_reduce'].W.data
    googlenet.inc5a.proj33.W.data = caffe_model['inception_5a/double3x3_reduce'].W.data
    googlenet.inc5a.poolp.W.data = caffe_model['inception_5a/pool_proj'].W.data

    # inc5b
    googlenet.inc5b.conv1.W.data = caffe_model['inception_5b/1x1'].W.data
    googlenet.inc5b.conv3.W.data = caffe_model['inception_5b/3x3'].W.data
    googlenet.inc5b.conv33a.W.data = caffe_model['inception_5b/double3x3a'].W.data
    googlenet.inc5b.conv33b.W.data = caffe_model['inception_5b/double3x3b'].W.data
    googlenet.inc5b.proj3.W.data = caffe_model['inception_5b/3x3_reduce'].W.data
    googlenet.inc5b.proj33.W.data = caffe_model['inception_5b/double3x3_reduce'].W.data
    googlenet.inc5b.poolp.W.data = caffe_model['inception_5b/pool_proj'].W.data

    googlenet.loss1_conv.W.data = caffe_model["loss1/conv"].W.data
    googlenet.loss1_fc1.W.data = caffe_model["loss1/fc"].W.data
    googlenet.loss2_conv.W.data = caffe_model["loss2/conv"].W.data
    googlenet.loss2_fc1.W.data = caffe_model["loss2/fc"].W.data

    serializers.save_npz('tuned_googlenetbn.npz', googlenet)
    print('Done')

if __name__ == '__main__':
    convert_caffe2chainer()
