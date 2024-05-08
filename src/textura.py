import cv2
import numpy as np
from matplotlib import pyplot as plt

def analizar_textura(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Especificar parámetros del filtro de Gabor
    kernel_size = 31
    theta = np.pi / 4
    sigma = 5
    lambd = 20.0
    gamma = 0.25

    # Crear el kernel de Gabor
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

    # Aplicar el filtro de Gabor a la imagen
    imagen_filtrada = cv2.filter2D(imagen_gris, cv2.CV_8UC3, kernel)

    # Mostrar la imagen original y la imagen filtrada
    plt.subplot(121), plt.imshow(imagen_gris, cmap='gray'), plt.title('Imagen Original')
    plt.subplot(122), plt.imshow(imagen_filtrada, cmap='gray'), plt.title('Imagen Filtrada con Gabor')
    plt.show()

# Ejemplo de uso
imagen_ejemplo = cv2.imread('/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/atardecer.jpg')
analizar_textura(imagen_ejemplo)