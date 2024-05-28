import cv2
import numpy as np
import os

for archivo in [x for x in os.listdir("/home/tille/Desktop/Tesis/code/img/") if x == "normal.jpeg"]:
    # Leer la imagen
    image = cv2.imread(f'/home/tille/Desktop/Tesis/code/img/{archivo}')
    # Leer la imagen en escala de grisesqq
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar bordes en la imagen usando Canny edge detector
    edges = cv2.Canny(gray, 170, 400)
#170
    # Definir el kernel para la operación de dilatación horizontal
    horizontal_kernel = np.ones((2, 3), np.uint8)

    # Dilatar los bordes detectados horizontalmente 3 veces
    dilated_edges_horizontal = cv2.dilate(edges, horizontal_kernel, iterations=3)

    height, width = image.shape[:2]
    min_line_length = int(width * 0.3)  # Longitud mínima relativa 0.5
    max_line_gap = int(width * 0.1)   # Espacio máximo entre líneas relativo 0.03

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


    # Encontrar líneas horizontales más largas
    longest_lines = []
    current_line = None
    max_length = 0

    # for line in lines:


    # for y in range(dilated_edges_horizontal.shape[0]):
    #     for x in range(dilated_edges_horizontal.shape[1]):
    #         if dilated_edges_horizontal[y, x] == 255:
    #             if current_line is None:
    #                 current_line = [(x, y)]
    #             else:
    #                 current_line.append((x, y))
    #         elif current_line is not None:
    #             if len(current_line) > max_length:
    #                 longest_lines = [current_line]
    #                 max_length = len(current_line)
    #             elif len(current_line) == max_length:
    #                 longest_lines.append(current_line)
    #             current_line = None
#1 
    # Encontrar coordenadas del pixel superior e inferior para cada línea más larga

    if horizontal_lines_in_range:
        line = horizontal_lines_in_range[0]
        x0 = 0
        xn = image.shape[1]-1
        x1, y1, x2, y2 = line[0]
        # Alta trigonometría
        y0 = int(y1 - ((x1-x0)*(y2-y1))/(x2-x1))
        yn = int(y0 + ((y2-y1)*(xn-x0))/(x2-x1))




    # Encontrar las coordenadas de la línea que abarca toda la región vertical
    # topmost_pixel = min(top_pixels, key=lambda x: x[0])[1]
    # bottommost_pixel = max(bottom_pixels, key=lambda x: x[0])[1]

    # # Calcular la posición vertical media entre las líneas detectadas
    # # midpoint_y = (topmost_pixel + bottommost_pixel) // 2

    # x,y,z = image.shape
    # print((0, topmost_pixel), (image.shape[1]-1, bottommost_pixel))

        # Dibujar la línea horizontal en el punto medio
        cv2.line(image, (x0, y0), (xn, yn), (0, 0, 255), 2)
        resized_image = cv2.resize(image, (500, 500))

        # Mostrar la imagen con la línea horizontal
        cv2.imshow('Horizontal Line', image)
        cv2.imshow('asdf Line', dilated_edges_horizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()