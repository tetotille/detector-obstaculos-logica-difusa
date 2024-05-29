import cv2
import numpy as np
import json


def encontrar_linea_mas_larga(contornos):
    """ Busca la línea más larga, con ella grafica el horizonte inclinado

    ### Args:
        contornos (cv2.contours): lineas detectadas por el filtro Canny

    ### Returns:
        best_line: retorna la mejor línea del horizonte
    """
    max_length = 0
    best_line = None

    for contorno in contornos:
        longitud = cv2.arcLength(contorno, True)

        if longitud > max_length:
            max_length = longitud
            best_line = cv2.approxPolyDP(contorno, 0.02 * longitud, True) # Ajustar la línea a la del contorno

    return best_line

if __name__ == "__main__":
    # Cargar la imagen
    img_path = json.load(open("config.json"))["img_path"] + "ruta-vista-inclinada-que-cruza-horizonte.jpg"
    image = cv2.imread(img_path)
    cv2.imshow('Imagen', image)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral adaptativo para mejorar el contraste
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(adaptive_threshold, 80000, 220000, apertureSize=7)
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    linea_mas_larga = encontrar_linea_mas_larga(contornos)

    # Dibujar la línea más larga encontrada
    if linea_mas_larga is not None:
        # Obtener los extremos de la línea
        x1, y1 = tuple(linea_mas_larga[0][0])
        x2, y2 = tuple(linea_mas_larga[-1][0])

        # Extender la línea hasta los bordes de la imagen con fórmulas matemáticas
        ancho, alto = image.shape[1], image.shape[0]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y1_nuevo = int(m * 0 + b)
        y2_nuevo = int(m * (ancho - 1) + b)

        cv2.line(image, (0, y1_nuevo), (ancho - 1, y2_nuevo), (0, 255, 0), 2)

    resized_image = cv2.resize(image, (600, 600))
    cv2.imshow('Imagen con Línea Más Larga Prolongada', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        # Aplicar un filtro de Canny para detectar bordes
        #edges = cv2.Canny(adaptive_threshold, 80000, 220000, apertureSize=7)
        

    # Cargar la imagen
    #image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/ruta-vista-inclinada-que-cruza-horizonte.jpg"







