import cv2
import numpy as np
import os

for archivo in os.listdir("C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img"):
    # Leer la imagen
    image = cv2.imread(f'C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/{archivo}')
    # Leer la imagen en escala de grisesq
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar bordes en la imagen usando Canny edge detector
    edges = cv2.Canny(gray, 170, 400)
#170
    # Definir el kernel para la operación de dilatación horizontal
    horizontal_kernel = np.ones((2, 3), np.uint8)

    # Dilatar los bordes detectados horizontalmente 3 veces
    dilated_edges_horizontal = cv2.dilate(edges, horizontal_kernel, iterations=3)

    height, width = image.shape[:2]
    min_line_length = int(height * 0.5)  # Longitud mínima relativa 0.5
    max_line_gap = int(height * 0.03)   # Espacio máximo entre líneas relativo 0.03

    # Hough Transform para detectar líneas
    lines = cv2.HoughLinesP(dilated_edges_horizontal, 1, np.pi/90, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)


    # Encontrar líneas horizontales más largas
    longest_lines = []
    current_line = None
    max_length = 0

    for y in range(dilated_edges_horizontal.shape[0]):
        for x in range(dilated_edges_horizontal.shape[1]):
            if dilated_edges_horizontal[y, x] == 255:
                if current_line is None:
                    current_line = [(x, y)]
                else:
                    current_line.append((x, y))
            elif current_line is not None:
                if len(current_line) > max_length:
                    longest_lines = [current_line]
                    max_length = len(current_line)
                elif len(current_line) == max_length:
                    longest_lines.append(current_line)
                current_line = None

    # Encontrar coordenadas del pixel superior e inferior para cada línea más larga
    top_pixels = []
    bottom_pixels = []

    for line in longest_lines:
        top_pixels.append(line[0])
        bottom_pixels.append(line[-1])


    # Encontrar las coordenadas de la línea que abarca toda la región vertical
    topmost_pixel = min(top_pixels, key=lambda x: x[1])[1]
    bottommost_pixel = max(bottom_pixels, key=lambda x: x[1])[1]

    # Calcular la posición vertical media entre las líneas detectadas
    midpoint_y = (topmost_pixel + bottommost_pixel) // 2

    x,y,z = image.shape

    # Dibujar la línea horizontal en el punto medio
    cv2.line(image, (0, midpoint_y), (image.shape[1]-1, midpoint_y), (0, 255, 0), 1)
    resized_image = cv2.resize(image, (300, 300))

    # Mostrar la imagen con la línea horizontal
    cv2.imshow('Horizontal Line', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()