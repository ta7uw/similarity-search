import chainer
import argparse
from func.dataset_function import dataset_label
from func.resize import resize
from func.compute_mean import compute_mean
from googlenetbn import GoogleNetBN
import numpy as np
from PIL import Image


def item_predict():
    parser = argparse.ArgumentParser(description="Predict item of image file ")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("pretrained_model")
    parser.add_argument("iamge")
    args = parser.parse_args()

    # Get the label data used for training.
    # It use for getting number of class when predicting
    b_names, _, label_name = dataset_label(args.dataset)

    model = GoogleNetBN(n_class=len(b_names))
    chainer.serializers.load_npz(args.pretained_model, model)
    mean = compute_mean(dataset_path=args.dataset, insize=model.insize).mean(axis=(1, 2))
    image = args.image
    image = Image.open(image)
    image = np.asarray(image, dtype=np.float32)
    image = resize(image, model.insize)
    image = image - mean[:, None, None]
    image = image.astype(np.float32)
    image *= (1.0 / 255.0)  # Scale to [0,1]

    chainer.config.train = False
    y = model.predict(image)
    y = y.data
    y = chainer.functions.softmax(y, axis=1)
    pred = b_names[int(y.argmax())]
    score = max(y.data)

    print("pred:", pred)
    print("score:", score)

if __name__ == '__main__':
    item_predict()