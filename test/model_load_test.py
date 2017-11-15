import chainer
import os, sys
print(os.getcwd())
sys.path.append(os.getcwd())

from googlenetbn import GoogleNetBN


def main():
    model = GoogleNetBN(n_class=10)
    chainer.serializers.load_npz("tuned_googlenetbn.npz", model)

if __name__ == '__main__':
    main()