import cv2
import numpy as np

def detectar_formas(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral binario
    _, umbral = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binarizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar contornos en la imagen original
    imagen_contornos = imagen.copy()
    cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

    # Mostrar la imagen original y la imagen con contornos
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Detección de Formas', imagen_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejemplo de uso
imagen_ejemplo = cv2.imread('/home/tille/Desktop/Tesis/code/img/barco.jpg')
detectar_formas(imagen_ejemplo)
