from googlenetbn import GoogleNetBN
from .dataset_function import dataset_label
import chainer
import pickle


def model2pkl():
    """
    Convert trained mdoel saved in .npz to .pkl
    Chainer Model save as pickle file
    """

    # Get the label data used for training.
    # It use for getting number of class when classifing
    b_names, label_name = dataset_label()

    # Define model for classification
    pretrained_model = "model-20epoch.npz"
    model = GoogleNetBN(n_class=len(b_names))
    chainer.serializers.load_npz(pretrained_model, model)

    with open("googlenetbn_tuned_model.pkl", "wb") as o:
        pickle.dump(model, o)

if __name__ == '__main__':
    model2pkl()

