import numpy as cp
import cv2
from os.path import dirname, abspath, join
from sys import argv
from src.utils import utils

def find_horizontal_line(imagen_cp):
    # Si la imagen está en color, convertirla a escala de grises
    if len(imagen_cp.shape) == 3:
        image = imagen_cp

    # Detectar bordes en la imagen usando edge detector
    edges = cp.array(cv2.imread("src/detector_horizonte/imagen_umbral.png", cv2.IMREAD_GRAYSCALE))
  # Asegúrate de que `process_image` también funcione con CuPy

    # Definir el kernel para la operación de dilatación horizontal
    horizontal_kernel = cp.ones((2, 3), cp.uint8)

    # Dilatar los bordes detectados horizontalmente 3 veces
    dilated_edges_horizontal = cv2.dilate(edges, horizontal_kernel, iterations=3)
    dilated_edges_horizontal = cp.array(dilated_edges_horizontal)

    height, width = imagen_cp.shape[:2]
    min_line_length = int(width * 0.3)  # Longitud mínima relativa 0.3
    max_line_gap = int(width * 0.1)     # Espacio máximo entre líneas relativo 0.1

    # Hough Transform para detectar líneas
    lines = cv2.HoughLinesP(dilated_edges_horizontal, 5, np.pi/60, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        lines = cp.array(lines)

    # Definir el rango vertical (20% a 80% de la altura de la imagen)
    vertical_range_top = int(height * 0.2)
    vertical_range_bottom = int(height * 0.8)

    # Función para filtrar líneas horizontales en el rango especificado
    def is_horizontal_and_in_range(line, threshold=5):
        x1, y1, x2, y2 = line[0]
        return abs(y1 - y2) <= threshold and vertical_range_top <= y1 <= vertical_range_bottom and vertical_range_top <= y2 <= vertical_range_bottom

    # Filtrar solo líneas horizontales en el rango especificado
    horizontal_lines_in_range = [line for line in lines if is_horizontal_and_in_range(line)] if lines is not None else []

    # Si se encuentra una línea horizontal en el rango
    if horizontal_lines_in_range:
        line = horizontal_lines_in_range[0]
        x0 = 0
        xn = image.shape[1] - 1
        x1, y1, x2, y2 = line[0]
        y0 = int(y1 - ((x1 - x0) * (y2 - y1)) / (x2 - x1))  # Calcular la coordenada y en x = 0
        yn = int(y0 + ((y2 - y1) * (xn - x0)) / (x2 - x1))  # Calcular la coordenada y en x = ancho de la imagen

        # Dibujar la línea horizontal en la imagen
        image = image
        cv2.line(image, (x0, y0), (xn, yn), (0, 0, 255), 2)
        image = cp.array(image)

        # Devolver la coordenada y del primer punto de la línea detectada
        return (height - y0), image
    else:
        return -1, image  # Si no se encuentra ninguna línea, devuelve -1 y la imagen original
    
def process_horizon_detection(image_path, new_width=200):
    # Leer la imagen desde el archivo
    image = cv2.imread(image_path)
    
    # Convertir la imagen a CuPy
    image_cupy = cp.array(image)
    
    # Obtener las dimensiones originales
    orig_height, orig_width, _ = image.shape
    
    # Calcular la nueva altura manteniendo la proporción
    new_height = int(orig_height * new_width / orig_width)
    
    # Redimensionar la imagen
    imagen_resized = utils.resize_image_bgr(image_cupy, (new_height, new_width))
    
    # Detectar la línea del horizonte
    y_horizonte, image_with_line = find_horizontal_line(imagen_resized)
    
    # Mostrar la coordenada del horizonte y devolver la imagen procesada
    print(f"Coordenada y del horizonte detectado: {y_horizonte}")
    return y_horizonte, image_with_line

# Ejemplo de uso
if __name__ == "__main__":
    filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
    y_horizonte, processed_image = process_horizon_detection(filename)