import numpy as np
try:
    import cupy as cp
except ImportError:
    print("cuda no está instalado.")
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
    lines = cv2.HoughLinesP(dilated_edges_horizontal, 5, cp.pi/60, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    #lines = probabilistic_hough_line_transform((dilated_edges_horizontal), 5, cp.pi/60, 100)
    if lines is not None:
        lines = cp.array(lines)

    # Definir el rango vertical (20% a 80% de la altura de la imagen)
    vertical_range_top = int(height * 0.2)
    vertical_range_bottom = int(height * 0.8)
    #print(lines)
    
    # Función para filtrar líneas horizontales en el rango especificado
    def is_horizontal_and_in_range(line, threshold=5):
        x1, y1, x2, y2 = line[0]
        return abs(y1 - y2) <= threshold and vertical_range_top <= y1 <= vertical_range_bottom and vertical_range_top <= y2 <= vertical_range_bottom
    
    # Filtrar solo líneas horizontales en el rango especificado
    horizontal_lines_in_range = [line for line in lines if is_horizontal_and_in_range(line)] if lines is not None else []
    """
    horizontal_lines_in_range = [
    line for line in lines
        
        if isinstance(line, cp.ndarray) and line.shape[0] == 4 and is_horizontal_and_in_range(line, vertical_range_top, vertical_range_bottom)
    ]
    """
    """
    if horizontal_lines_in_range:
        line = horizontal_lines_in_range[0]

        print("linea 0", line)  # Imprimir el contenido del array de CuPy directamente

        x0 = cp.int32(0)  # Usar CuPy para inicializar
        xn = cp.int32(image.shape[1] - 1)  # Ancho de la imagen - 1
        # Descomponer la línea directamente usando CuPy
        x1, y1, x2, y2 = map(float, line.flatten().get())  # Convierte a float y usa `.get()` para asegurarte de que son escalares de Python

    # Calcular coordenadas y como escalares
        y0 = int(y1 - ((x1 - x0) * (y2 - y1)) / (x2 - x1))  # Calcula y0
        yn = int(y0 + ((y2 - y1) * (xn - x0)) / (x2 - x1))  # Calcula yn
        y0 = cp.int32(y0)
        yn = cp.int32(yn)
        line = draw_line_cupy(image, x0, y0, xn, yn, (0, 0, 255))
        # Convertir las coordenadas a enteros
        return (height - y0), image
    else:
        return -1, image  # Si no se encuentra ninguna línea, devuelve -1 y la imagen original
    
    """
    # Si se encuentra una línea horizontal en el rango
    if horizontal_lines_in_range:
        line = horizontal_lines_in_range[0]

        #print("linea 0", line[0])
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
        return -1, image  # Si no se encuentra ninguna línea, devuelve -1 y la imagen original"""
    
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

def probabilistic_hough_line_transform(edges_dilated, rho=1, theta=np.pi/180, threshold=100):
    """
    Implementación de la Transformada de Hough Probabilística utilizando CuPy para procesamiento con GPU.

    Parámetros:
    - edges_dilated: Imagen de entrada en escala de grises (imagen de bordes dilatada).
    - rho: Resolución de r en píxeles.
    - theta: Resolución de θ en radianes.
    - threshold: Número mínimo de intersecciones necesarias para detectar una línea.
    - min_line_length: Longitud mínima de una línea.
    - max_line_gap: Máxima separación entre segmentos para considerarlos como una sola línea.

    Retorna:
    - lines: Lista de líneas detectadas, cada línea representada como un array [x0, y0, x1, y1].
    """
    # Paso 2: Crear un acumulador con resoluciones especificadas para r y θ
    height, width = edges_dilated.shape
    minimo=cp.minimum(height, width)
    diag_len = int(cp.sqrt(width**2 + height**2))  # Longitud máxima posible de una línea en la imagen
    rhos = cp.arange(-diag_len, diag_len, rho)
    thetas = cp.arange(0, np.pi, theta)
    accumulator = cp.zeros((len(rhos), len(thetas)), dtype=cp.int32)

    # Paso 3: Obtener coordenadas de los bordes detectados
    y_indices, x_indices = cp.nonzero(cp.asarray(edges_dilated))  # Coordenadas de los bordes en CuPy

    # Calcular r para todos los puntos de borde en una sola operación
    x = x_indices[:, cp.newaxis]  # Asegurarse de que x tenga una forma adecuada (num_edges, 1)
    y = y_indices[:, cp.newaxis]  # Asegurarse de que y tenga una forma adecuada (num_edges, 1)

    # Calcular r para todos los ángulos theta
    cos_t = cp.cos(thetas)  # Shape: (num_thetas,)
    sin_t = cp.sin(thetas)  # Shape: (num_thetas)

    # Expandir dimensiones para broadcasting
    r = x * cos_t + y * sin_t  # Shape: (num_edges, num_thetas)

    # Convertir rho a índices en el acumulador (considerando rhos como referencia)
    r_indices = cp.asarray(((r - rhos[0]) / rho).round().astype(cp.int32))  # Shape: (num_edges, num_thetas)

    # Verificar que los índices de rho estén dentro del rango válido
    valid_mask = (r_indices >= 0) & (r_indices < len(rhos))

    for t_index in range(len(thetas)):
        # Seleccionar los índices válidos de rho para el ángulo actual
        valid_r_indices = r_indices[:, t_index][valid_mask[:, t_index]]

        # Verificar que valid_r_indices no esté vacío antes de usarlo
        if valid_r_indices.size > 0:
            cp.add.at(accumulator, (valid_r_indices, t_index), 1)

    # Paso 4: Detectar picos en el acumulador
    peak_indices = cp.argwhere(accumulator >= threshold)

    # Convertir los picos en coordenadas rho y theta
    detected_rhos = rhos[peak_indices[:, 0]]
    detected_thetas = thetas[peak_indices[:, 1]]

    # Paso 5: Convertir los picos detectados a líneas (x0, y0, x1, y1)
    lines = cp.empty((0, 4), dtype=cp.int32)
    for rho, theta in zip(detected_rhos, detected_thetas):
        minim=float(minimo)
        a = cp.cos(theta)
        b = cp.sin(theta)
        x0 = cp.asarray(a * rho)
        y0 = cp.asarray(b * rho)
        x1 = cp.asarray(x0 + minim * cp.asarray(-b))
        y1 = cp.asarray(y0 + minim *a)
        # Guardar la línea detectada
        new_line = cp.array([x0, y0, x1, y1], dtype=cp.int32)
        lines = cp.vstack([lines, new_line])

    return lines

# Ejemplo de uso:
def is_horizontal_and_in_range(line, vertical_range_top, vertical_range_bottom, threshold=5):
    """
    Verifica si la línea es horizontal y si se encuentra dentro del rango vertical especificado.
    
    Parámetros:
    - line: array de CuPy con las coordenadas [x1, y1, x2, y2] de la línea.
    - vertical_range_top: límite superior del rango vertical.
    - vertical_range_bottom: límite inferior del rango vertical.
    - threshold: diferencia máxima permitida en las coordenadas y para considerar la línea como horizontal.
    
    Retorna:
    - La línea si es horizontal y está en el rango, de lo contrario None.
    """
    x1, y1, x2, y2 = line
    #print(line)

    # Verificar si la línea es horizontal según el umbral especificado
    is_horizontal = cp.abs(y1 - y2) <= threshold

    # Retornar la línea si cumple ambas condiciones, de lo contrario None
    return bool(is_horizontal)

def bresenham(x0, y0, xn, yn):
    """Genera los puntos de la línea usando el algoritmo de Bresenham."""
    points = []
    dx = xn - x0
    dy = yn - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        err = dx / 2.0
        while x0 != xn:
            points.append((x0, y0))
            err -= dy
            if err < 0:
                y0 += sy
                err += dx
            x0 += sx
    else:
        err = dy / 2.0
        while y0 != yn:
            points.append((x0, y0))
            err -= dx
            if err < 0:
                x0 += sx
                err += dy
            y0 += sy
    points.append((xn, yn))
    return points

def draw_line_cupy(image, x0, y0, xn, yn, color):
    """Dibuja una línea en la imagen usando CuPy."""
    # Asegúrate de que la imagen esté en formato (H, W, C) y en tipo uint8
    points = bresenham(x0, y0, xn, yn)
    for x, y in points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Verifica los límites
            image[y, x] = cp.array(color, dtype=image.dtype)

# Ejemplo de uso
if __name__ == "__main__":
    filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
    y_horizonte, processed_image = process_horizon_detection(filename)