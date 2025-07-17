import cv2
import numpy as np
from os.path import abspath,dirname,join
from sys import argv

def detectar_horizonte(image):
    """ Halla la línea del horizonte a través de las colisiones de dos gradientes de color.

    ### Args:
        image (Image): Imagen de el agua

    ### Returns:
        image, transition_index: retorna la imagen dibujada con la línea del horizonte y la línea completa del horizonte
    """

    # Filtros normales
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = np.gradient(blurred_image, axis=0)
    smoothed_gradient = np.convolve(np.mean(gradient, axis=1), np.ones(15)/15, mode='same')

    # Encontrar el punto de transición más significativo y dibujar la línea de horizonte
    transition_index = np.argmax(np.abs(smoothed_gradient))
    if transition_index is not None:
        cv2.line(image, (0, transition_index), (image.shape[1], transition_index), (0, 255, 0), thickness=2)

    return image, transition_index

def detect_horizon(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta del archivo y asegúrate de que el archivo exista.")

    # Detectar el horizonte
    horizonte_image, horizon_y = detectar_horizonte(image)

    # Mostrar la imagen con el horizonte detectado
    resized_image = cv2.resize(horizonte_image, (600, 600))  # Aumentar el tamaño para mejor visualización
    cv2.imshow('Horizon Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ruta a la imagen
    if len(argv)>1:
        filename = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))),"img/IMG_6830.jpeg")

    # image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/ruta-vista-inclinada-que-cruza-horizonte.jpg"
    detect_horizon(filename)



