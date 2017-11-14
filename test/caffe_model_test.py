from chainer.links.caffe import CaffeFunction


def main():
    CAFFEMODEL_FN = "googlenet.caffemodel"
    model = CaffeFunction(CAFFEMODEL_FN)

    for layer in model.layers:
        print(layer)
        print("--------------")

    for split in model.split_map:
        print(split)
        print("--------------")

    for forward in model.forwards:
        print(forward)
        print("--------------")


if __name__ == '__main__':
    main()