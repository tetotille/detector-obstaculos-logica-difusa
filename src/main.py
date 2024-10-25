import cupy as cp

from os.path import join, dirname, abspath

from utils import read_image


if __name__ == "__main__":
    filename = dirname(dirname(__file__)) + "/assets/images/akaso1.jpeg"

    image:cp.array  = read_image(filename, 256)

    