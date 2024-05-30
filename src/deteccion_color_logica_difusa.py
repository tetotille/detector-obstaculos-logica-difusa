import cv2
import numpy as np
import matplotlib.pyplot as plt
from fuzzylogic.classes import Domain, Set
from fuzzylogic.functions import bounded_sigmoid, triangular
import json

def detectar_color(imagen, rango_color):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    rango_bajo = np.array(rango_color[0])
    rango_alto = np.array(rango_color[1])
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
    return resultado, mascara

def calcular_centroide(mascara):
    M = cv2.moments(mascara)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

def aplicar_logica_difusa_posicion(centroide, dimensiones, mostrar_grafica=False):
    x, y = centroide
    ancho, alto = dimensiones

    pos_x = Domain("x", 0, ancho)
    pos_y = Domain("y", 0, alto)

    pos_x.left = triangular(0, ancho/2)
    pos_x.center = triangular(ancho/4, (3*ancho)/4)
    pos_x.right = triangular(ancho/2,  ancho)

    pos_y.top = triangular(0, alto/2)
    pos_y.bottom = triangular(alto/2, alto)

    grado_izq = pos_x.left(x)
    grado_centro = pos_x.center(x)
    grado_der = pos_x.right(x)

    grado_arriba = pos_y.top(y)
    grado_abajo = pos_y.bottom(y)

    pos_x_res = ""
    pos_y_res = ""

    if grado_izq >= grado_centro and grado_centro!=0.00:
        pos_x_res = "centrado hacia la izquierda"
    elif grado_der >= grado_centro and grado_centro!=0.00:
        pos_x_res = "centrado hacia la derecha"
    elif grado_izq > grado_centro and grado_izq > grado_der:
        pos_x_res = "a la izquierda"
    elif grado_centro > grado_izq and grado_centro > grado_der:
        pos_x_res = "en el centro"
    else:
        pos_x_res = "a la derecha"

    if grado_arriba >=  grado_abajo:
        pos_y_res = "arriba"
    else:
        pos_y_res = "abajo"

    if mostrar_grafica:
        x_vals = np.arange(0, ancho, 1)
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, [pos_x.left(val) for val in x_vals], 'b', linewidth=1.5, label='Izquierda')
        plt.plot(x_vals, [pos_x.center(val) for val in x_vals], 'g', linewidth=1.5, label='Centro')
        plt.plot(x_vals, [pos_x.right(val) for val in x_vals], 'r', linewidth=1.5, label='Derecha')
        plt.title('Funciones de Pertenencia para la Posición Horizontal')
        plt.xlabel('Coordenada X')
        plt.ylabel('Grado de Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

        y_vals = np.arange(0, alto, 1)
        plt.figure(figsize=(10, 5))
        plt.plot(y_vals, [pos_y.top(val) for val in y_vals], 'b', linewidth=1.5, label='Arriba')
        plt.plot(y_vals, [pos_y.bottom(val) for val in y_vals], 'r', linewidth=1.5, label='Abajo')
        plt.title('Funciones de Pertenencia para la Posición Vertical')
        plt.xlabel('Coordenada Y')
        plt.ylabel('Grado de Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

    return f"{pos_y_res} {pos_x_res}", grado_izq, grado_centro, grado_der

def aplicar_logica_difusa_pixeles(num_pixeles, total_pixeles, mostrar_grafica=False):
    pocos_umbral = 0.15 * total_pixeles
    muchos_umbral = 0.10 * total_pixeles

    pixeles = Domain("pixeles", 0, total_pixeles)

    pixeles.pocos = bounded_sigmoid(0, muchos_umbral, inverse=True)
    pixeles.muchos = bounded_sigmoid(pocos_umbral, total_pixeles, inverse=True)

    grado_pocos = pixeles.pocos(num_pixeles)
    grado_muchos = pixeles.muchos(num_pixeles)

    if mostrar_grafica:
        x = np.arange(0, total_pixeles + 1, 1)
        plt.figure(figsize=(10, 5))
        plt.plot(x, [pixeles.pocos(val) for val in x], 'b', linewidth=1.5, label='Pocos píxeles')
        plt.plot(x, [pixeles.muchos(val) for val in x], 'r', linewidth=1.5, label='Muchos píxeles')
        plt.title('Funciones de Pertenencia para la Cantidad de Píxeles')
        plt.xlabel('Número de Píxeles')
        plt.ylabel('Grado de Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Número de píxeles: {num_pixeles}")
    print(f"Grado de pertenencia a pocos píxeles: {grado_pocos}")
    print(f"Grado de pertenencia a muchos píxeles: {grado_muchos}")

    if grado_muchos > grado_pocos:
        return "Es un objeto"
    else:
        return "No es un objeto"

if __name__ == "__main__":
    image_path = json.load(open("config.json"))["img_path"] + "barco.jpg"
    imagen_camara = cv2.imread(image_path)
    scale_percent = 40
    width = int(imagen_camara.shape[1] * scale_percent / 100)
    height = int(imagen_camara.shape[0] * scale_percent / 100)
    dim = (width, height)
    imagen_camara = cv2.resize(imagen_camara, dim, interpolation=cv2.INTER_AREA)

    rango_color = [[0, 100, 0], [100, 255, 200]]

    resultado, mascara = detectar_color(imagen_camara, rango_color)
    centroide = calcular_centroide(mascara)
    print("Centroide:", centroide)

    posicion, grado_izq, grado_centro, grado_der = aplicar_logica_difusa_posicion(centroide, (width, height), mostrar_grafica=True)
    print("Posición:", posicion)
    print(f"Grado de pertenencia a la izquierda: {grado_izq}")
    print(f"Grado de pertenencia en el centro: {grado_centro}")
    print(f"Grado de pertenencia a la derecha: {grado_der}")

    num_pixeles = cv2.countNonZero(mascara)
    total_pixeles = width * height
    print("Número de píxeles:", num_pixeles)

    es_objeto = aplicar_logica_difusa_pixeles(num_pixeles, total_pixeles, mostrar_grafica=True)
    print("¿Es un objeto?:", es_objeto)
