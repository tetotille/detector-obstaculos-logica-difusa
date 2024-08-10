import cv2
import numpy as np
from contour_detection import process_image

def find_horizontal_line(imagen):
    # Si la imagen está en color, convertirla a escala de grises
    if len(imagen.shape) == 3:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detectar bordes en la imagen usando Canny edge detector
    edges = process_image(imagen)

    # Definir el kernel para la operación de dilatación horizontal
    horizontal_kernel = np.ones((2, 3), np.uint8)

    # Dilatar los bordes detectados horizontalmente 3 veces
    dilated_edges_horizontal = cv2.dilate(edges, horizontal_kernel, iterations=3)
    x, y, a = imagen.shape
    image = cv2.resize(imagen, (200, int(x*200/y)))

    height, width = image.shape[:2]
    min_line_length = int(width * 0.3)  # Longitud mínima relativa 0.3
    max_line_gap = int(width * 0.1)     # Espacio máximo entre líneas relativo 0.1

    # Hough Transform para detectar líneas
    lines = cv2.HoughLinesP(dilated_edges_horizontal, 5, np.pi/60, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Definir el rango vertical (20% a 80% de la altura de la imagen)
    vertical_range_top = int(height * 0.2)
    vertical_range_bottom = int(height * 0.8)

    # Función para filtrar líneas horizontales en el rango especificado
    def is_horizontal_and_in_range(line, threshold=5):
        x1, y1, x2, y2 = line[0]
        return abs(y1 - y2) <= threshold and vertical_range_top <= y1 <= vertical_range_bottom and vertical_range_top <= y2 <= vertical_range_bottom

    # Filtrar solo líneas horizontales en el rango especificado
    horizontal_lines_in_range = [line for line in lines if is_horizontal_and_in_range(line)]

    # Si se encuentra una línea horizontal en el rango
    if horizontal_lines_in_range:
        line = horizontal_lines_in_range[0]
        x0 = 0
        xn = image.shape[1] - 1
        x1, y1, x2, y2 = line[0]
        y0 = int(y1 - ((x1 - x0) * (y2 - y1)) / (x2 - x1))  # Calcular la coordenada y en x = 0
        yn = int(y0 + ((y2 - y1) * (xn - x0)) / (x2 - x1))  # Calcular la coordenada y en x = ancho de la imagen

        # Dibujar la línea horizontal en la imagen
        cv2.line(image, (x0, y0), (xn, yn), (0, 0, 255), 2)
        cv2.imshow('horizonte', image)
        
        # Devolver la coordenada y del primer punto de la línea detectada
        return (height-y0), image
    else:
        return -1, image  # Si no se encuentra ninguna línea, devuelve -1 y la imagen original
