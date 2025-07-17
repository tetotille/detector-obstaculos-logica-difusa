from os.path import join, dirname, abspath
from src.utils import read_image


if __name__ == "__main__":
    filename = join(dirname(dirname(dirname(abspath(__file__)))), "assets/images/barco.jpg")
    image = read_image(filename)
    
