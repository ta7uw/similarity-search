import numpy as np
import os, sys

sys.path.append(os.getcwd())


def main():
    model = np.load("tuned_googlenetbn.npz")
    for x in model:
        print(x)


if __name__ == '__main__':
    main()