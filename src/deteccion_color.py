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
    scale_percent = 40 # percent of original size
    width = int(imagen_camara.shape[1] * scale_percent / 100)
    height = int(imagen_camara.shape[0] * scale_percent / 100)
    dim = (width, height)
    imagen_camara = cv2.resize(imagen_camara,dim,interpolation = cv2.INTER_AREA)
    # Definir el rango de color (ajustar según el objeto que estás buscando)
    rango_color = [[0, 100, 0], [100, 255, 200]]  # Rango bajo, Rango alto
    
    hsv = cv2.cvtColor(imagen_camara, cv2.COLOR_BGR2HSV)
    
    # Definir un rango de colores en formato HSV
    rango_bajo = np.array(rango_color[0])
    rango_alto = np.array(rango_color[1])
    
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)
    
    resultado = cv2.bitwise_and(imagen_camara, imagen_camara, mask=mascara)
    
    # Horizontally concatenate the 2 images
    img3 = cv2.hconcat([imagen_camara,resultado])
    
    cv2.imshow('Imagen hsv', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()