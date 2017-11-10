import pickle
from chainer.links.caffe import CaffeFunction


def main():
    CAFFEMODEL_FN = "googlenet.caffemodel"
    model = CaffeFunction(CAFFEMODEL_FN)
    for layer in model.layers:
        print(layer)


if __name__ == '__main__':
    main()