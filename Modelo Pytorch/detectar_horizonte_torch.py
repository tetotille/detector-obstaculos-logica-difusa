import torch
import torch.nn.functional as F
import cv2
import numpy as np

def find_horizontal_line(image_tensor):
    # Si la imagen está en color, convertirla a escala de grises
    if len(image_tensor.shape) == 3:
        image_tensor = torch.mean(image_tensor, dim=2).unsqueeze(2)  # Convertir a escala de grises

    # Leer la imagen de bordes ya procesada
    edges = torch.tensor(cv2.imread("imagen_umbral.png", cv2.IMREAD_GRAYSCALE), dtype=torch.float32)

    # Definir el kernel para la operación de dilatación horizontal
    horizontal_kernel = torch.ones((1, 1, 2, 3), dtype=torch.float32)  # Cambiar a 4D para conv2d

    # Dilatar los bordes detectados horizontalmente 3 veces
    for _ in range(3):
        # Añadir una dimensión para que el tensor tenga forma [batch_size, channels, height, width]
        dilated_edges_horizontal = F.conv2d(edges.unsqueeze(0).unsqueeze(0), horizontal_kernel, padding=(1, 1))
        
        # Tomar el máximo entre el resultado convolucionado y la imagen original
        edges = torch.maximum(edges, dilated_edges_horizontal.squeeze(0).squeeze(0))

    height, width = image_tensor.shape[:2]
    min_line_length = int(width * 0.3)  # Longitud mínima relativa 0.3
    max_line_gap = int(width * 0.1)     # Espacio máximo entre líneas relativo 0.1

    # Hough Transform para detectar líneas
    lines = cv2.HoughLinesP(edges.numpy(), 5, np.pi / 60, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        lines = torch.tensor(lines)

    # Definir el rango vertical (20% a 80% de la altura de la imagen)
    vertical_range_top = int(height * 0.2)
    vertical_range_bottom = int(height * 0.8)

    # Función para filtrar líneas horizontales en el rango especificado
    def is_horizontal_and_in_range(line, threshold=5):
        x1, y1, x2, y2 = line[0]
        return abs(y1 - y2) <= threshold and vertical_range_top <= y1 <= vertical_range_bottom and vertical_range_top <= y2 <= vertical_range_bottom
    
    # Filtrar solo líneas horizontales en el rango especificado
    horizontal_lines_in_range = [line for line in lines if is_horizontal_and_in_range(line)] if lines is not None else []

    if horizontal_lines_in_range:
        line = horizontal_lines_in_range[0]
        print("Línea encontrada:", line)  # Imprimir el contenido de la línea

    return horizontal_lines_in_range

import torch

def is_horizontal_and_in_range(line, vertical_range_top, vertical_range_bottom, threshold=5):
    """
    Verifica si la línea es horizontal y si se encuentra dentro del rango vertical especificado.
    
    Parámetros:
    - line: tensor de PyTorch con las coordenadas [x1, y1, x2, y2] de la línea.
    - vertical_range_top: límite superior del rango vertical.
    - vertical_range_bottom: límite inferior del rango vertical.
    - threshold: diferencia máxima permitida en las coordenadas y para considerar la línea como horizontal.
    
    Retorna:
    - La línea si es horizontal y está en el rango, de lo contrario None.
    """
    x1, y1, x2, y2 = line

    # Verificar si la línea es horizontal según el umbral especificado
    is_horizontal = torch.abs(y1 - y2) <= threshold

    # Comprobar si las coordenadas y de la línea están dentro del rango especificado
    is_within_vertical_range = (vertical_range_top <= y1 <= vertical_range_bottom) and (vertical_range_top <= y2 <= vertical_range_bottom)

    # Retornar la línea si cumple ambas condiciones, de lo contrario None
    return line if is_horizontal and is_within_vertical_range else None

import torch

def bresenham(x0, y0, xn, yn):
    """Genera los puntos de la línea usando el algoritmo de Bresenham en PyTorch."""
    points = []
    x0, y0, xn, yn = map(torch.tensor, (x0, y0, xn, yn))
    
    dx = xn - x0
    dy = yn - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        err = dx / 2.0
        while x0 != xn:
            points.append((x0.item(), y0.item()))  # Convertir a Python float antes de añadir
            err -= dy
            if err < 0:
                y0 += sy
                err += dx
            x0 += sx
    else:
        err = dy / 2.0
        while y0 != yn:
            points.append((x0.item(), y0.item()))  # Convertir a Python float antes de añadir
            err -= dx
            if err < 0:
                x0 += sx
                err += dy
            y0 += sy

    points.append((xn.item(), yn.item()))  # Convertir a Python float antes de añadir
    return points

def draw_line_pytorch(image, x0, y0, xn, yn, color):
    """Dibuja una línea en la imagen usando PyTorch."""
    # Asegúrate de que la imagen esté en formato (H, W, C) y en tipo uint8
    points = bresenham(x0, y0, xn, yn)
    
    for x, y in points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Verifica los límites
            image[y, x] = torch.tensor(color, dtype=image.dtype)  # Asigna el color