import numpy as np
import cv2
import cupy as cp
from cupyx import fallback_mode


hacer_mascara_kernel = cp.RawKernel(open("kernels/hacer_mascara_kernel.cu").read(), "hacer_mascara_kernel")

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

def hacer_mascara(image3, mask):
    height, width, channels = image3.shape
    mask_height, mask_width = mask.shape

    # Crear imágenes para almacenar los resultados
    contour_image = cp.zeros((height, width, channels), dtype=cp.uint8)
    rgba_image = cp.zeros((height, width, 4), dtype=cp.uint8)

    # Configuración de bloques e hilos para el kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Ejecutar el kernel
    hacer_mascara_kernel(
        blocks_per_grid, threads_per_block,
        (image3, mask, contour_image, rgba_image, width, height, channels, mask_width, mask_height)
    )

    return contour_image, rgba_image

def read_image(image_path:str,new_width:int,**params) -> tuple[np.array,cp.array]:
    """
    Lee una imagen de un archivo y la convierte en un array de CuPy.
    Args:
        image_path: La ruta del archivo de imagen.
    Returns:
        La imagen como una tupla de arrays de NumPy y CuPy.
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
    with fallback_mode():
        return image,cp.asarray(image)


if __name__ == "__main__":
    # Leer la imagen y convertirla a un array de CuPy
    image_path = "/home/tesis_liz_tille/detector-obstaculos-logica-difusa/assets/images/akaso1.jpeg"
    image = read_image(image_path, 256)
    cv2.imwrite("result.jpg", cp.asnumpy(image))