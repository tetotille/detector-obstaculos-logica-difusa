import cv2
import numpy as np

# Función para procesar la imagen y detectar un rango de colores
def detectar_color(imagen, rango_color):
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Definir un rango de colores en formato HSV
    rango_bajo = np.array(rango_color[0])
    rango_alto = np.array(rango_color[1])

    # Crear una máscara utilizando el rango de colores
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    return resultado

if __name__ == "__main__":
    # Capturar imagen de la cámara (reemplazar con tu propia lógica para obtener imágenes)
    imagen_camara = cv2.imread('/home/tille/Desktop/Tesis/code/img/barco.jpg')

    # Definir el rango de color (ajustar según el objeto que estás buscando)
    rango_color_amarillo = [[30, 100, 100], [60, 255, 255]]  # Rango bajo, Rango alto

    # Aplicar la función de detección de color
    resultado_color = detectar_color(imagen_camara, rango_color_amarillo)
    # Mostrar la imagen original y el resultado
    cv2.imshow('Imagen Original', imagen_camara)
    cv2.imshow('Resultado Detección de Color', resultado_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()