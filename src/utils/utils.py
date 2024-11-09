import numpy as np
import cv2
import cupy as cp
from matplotlib import pyplot as plt


class FuzzyImage:
    def __init__(self,img_path:str):
        self.image = cv2.imread(img_path)
        if self.image is None:
            raise ValueError("No se pudo cargar la imagen. Verifique la ruta del archivo y asegúrese de que el archivo exista.")
        self.image = self.__resize_image(256)
    
    def __resize_image(self,new_width:int):
        orig_height, orig_width, channels = self.image.shape
        new_height = int(orig_height * new_width / orig_width)
        return cv2.resize(self.image, (new_width, new_height))


def crop_horizontal(imagen, indice_vertical):
    """
    Recorta una imagen a color horizontalmente en un índice dado.
    Args:
        imagen: Una imagen a color en formato NumPy.
        indice_vertical: El índice vertical donde se realizará el recorte.
    Returns:
        Una tupla que contiene dos imágenes: la parte superior y la parte inferior.
    """

    if indice_vertical < 0 or indice_vertical >= imagen.shape[0]:
        raise ValueError("El índice vertical está fuera de los límites de la imagen.")
    parte_superior = imagen[:indice_vertical, :]
    parte_inferior = imagen[indice_vertical:, :]
    return parte_superior, parte_inferior

def read_image(image_path:str,new_width:int,new_height:int=None,**params) -> cp.array:
    """
    Lee una imagen de un archivo y la convierte en un array de CuPy.
    Args:
        image_path: La ruta del archivo de imagen.
    Returns:
        La imagen como un array de CuPy.
    """
    grayscale = params.get("grayscale", False)
    normalize = params.get("normalize", False)
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifique la ruta del archivo y asegúrese de que el archivo exista.")
    
    # Redimensionar la imagen al nuevo ancho
    if new_height is not None:
        new_height = int(image.shape[0] * new_width / image.shape[1])
    image = cv2.resize(image, (new_width, new_height))
    if normalize:
        image = image / 255.0
    return cp.asarray(image)


if __name__ == "__main__":
    # Leer la imagen y convertirla a un array de CuPy
    image_path = "/home/tesis_liz_tille/detector-obstaculos-logica-difusa/assets/images/akaso1.jpeg"
    image = read_image(image_path, 256)
    cv2.imwrite("result.jpg", cp.asnumpy(image))