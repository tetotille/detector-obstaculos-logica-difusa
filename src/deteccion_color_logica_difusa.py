import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import json

def detectar_color(imagen, rango_color):
    """Busca obstáculos verificando que los colores en la imagen no están en un rango
    conocido que corresponde al paisaje.
    Para la búsqueda se utilizaron paisajes con brillo, con atardeceres, normales.
    Se buscó el histograma de colores de cada uno y se comparó.

    Args:
        imagen (cv2.Image): Imagen que se desea analizar
        rango_color (): Rango de colores correspondiente a los obstáculos

    Returns:
        resultado: Retorna la imagen del color
        mascara: Retorna la máscara binaria del color detectado
    """
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Definir un rango de colores en formato HSV
    rango_bajo = np.array(rango_color[0])
    rango_alto = np.array(rango_color[1])
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    return resultado, mascara

def calcular_centroide(mascara):
    """Calcula el centroide de la máscara de píxeles detectados.

    Args:
        mascara (np.array): Máscara binaria de píxeles detectados

    Returns:
        tuple: Coordenadas del centroide (x, y)
    """
    # Calcular los momentos de la máscara
    M = cv2.moments(mascara)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

def aplicar_logica_difusa_posicion(centroide, dimensiones, mostrar_grafica=False):
    """Aplica lógica difusa para determinar la posición del objeto en la imagen.
    
    Args:
        centroide (tuple): Coordenadas del centroide (x, y).
        dimensiones (tuple): Dimensiones de la imagen (ancho, alto).
        mostrar_grafica (bool): Si es True, muestra la gráfica de las funciones de pertenencia.
        
    Returns:
        str: Resultado de la lógica difusa (posición del objeto).
    """
    x, y = centroide
    ancho, alto = dimensiones

    # Definir las funciones de pertenencia para la posición horizontal
    x_vals = np.arange(0, ancho, 1)
    izq = fuzz.trimf(x_vals, [0, 0, ancho/3])
    centro = fuzz.trimf(x_vals, [ancho/3, ancho/2, 2*ancho/3])
    der = fuzz.trimf(x_vals, [2*ancho/3, ancho, ancho])

    # Definir las funciones de pertenencia para la posición vertical
    y_vals = np.arange(0, alto, 1)
    arriba = fuzz.trimf(y_vals, [0, 0, alto/3])
    medio = fuzz.trimf(y_vals, [alto/3, alto/2, 2*alto/3])
    abajo = fuzz.trimf(y_vals, [2*alto/3, alto, alto])

    if mostrar_grafica:
        # Graficar las funciones de pertenencia para la posición horizontal
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, izq, 'b', linewidth=1.5, label='Izquierda')
        plt.plot(x_vals, centro, 'g', linewidth=1.5, label='Centro')
        plt.plot(x_vals, der, 'r', linewidth=1.5, label='Derecha')
        plt.title('Funciones de Pertenencia para la Posición Horizontal')
        plt.xlabel('Coordenada X')
        plt.ylabel('Grado de Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Graficar las funciones de pertenencia para la posición vertical
        plt.figure(figsize=(10, 5))
        plt.plot(y_vals, arriba, 'b', linewidth=1.5, label='Arriba')
        plt.plot(y_vals, medio, 'g', linewidth=1.5, label='Medio')
        plt.plot(y_vals, abajo, 'r', linewidth=1.5, label='Abajo')
        plt.title('Funciones de Pertenencia para la Posición Vertical')
        plt.xlabel('Coordenada Y')
        plt.ylabel('Grado de Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Calcular los grados de pertenencia para la posición horizontal
    grado_izq = fuzz.interp_membership(x_vals, izq, x)
    grado_centro = fuzz.interp_membership(x_vals, centro, x)
    grado_der = fuzz.interp_membership(x_vals, der, x)

    # Calcular los grados de pertenencia para la posición vertical
    grado_arriba = fuzz.interp_membership(y_vals, arriba, y)
    grado_medio = fuzz.interp_membership(y_vals, medio, y)
    grado_abajo = fuzz.interp_membership(y_vals, abajo, y)

    # Combinar los grados de pertenencia para determinar la posición final
    pos_x = ""
    pos_y = ""

    if grado_izq >= grado_centro and grado_izq >= grado_der:
        pos_x = "a la izquierda"
    elif grado_centro >= grado_izq and grado_centro >= grado_der:
        pos_x = "en el centro"
    else:
        pos_x = "a la derecha"

    if grado_arriba >= grado_medio and grado_arriba >= grado_abajo:
        pos_y = "arriba"
    elif grado_medio >= grado_arriba and grado_medio >= grado_abajo:
        pos_y = "en el medio"
    else:
        pos_y = "abajo"

    return f"{pos_y} {pos_x}"

def aplicar_logica_difusa_pixeles(num_pixeles, mostrar_grafica=False):
    """Aplica lógica difusa para determinar si un conjunto de píxeles representa un objeto.
    
    Args:
        num_pixeles (int): Cantidad de píxeles detectados del color específico.
        mostrar_grafica (bool): Si es True, muestra la gráfica de las funciones de pertenencia.
        
    Returns:
        str: Resultado de la lógica difusa ("Es un objeto" o "No es un objeto").
    """
    x = np.arange(0, 10001, 1)
    pixeles_pocos = fuzz.trimf(x, [0, 0, 1000])
    pixeles_muchos = fuzz.trimf(x, [1000, 10000, 10000])

    if mostrar_grafica:
        # Graficar las funciones de pertenencia
        plt.figure(figsize=(10, 5))
        plt.plot(x, pixeles_pocos, 'b', linewidth=1.5, label='Pocos píxeles')
        plt.plot(x, pixeles_muchos, 'r', linewidth=1.5, label='Muchos píxeles')
        plt.title('Funciones de Pertenencia para la Cantidad de Píxeles')
        plt.xlabel('Número de Píxeles')
        plt.ylabel('Grado de Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Calcular los grados de pertenencia
    grado_pocos = fuzz.interp_membership(x, pixeles_pocos, num_pixeles)
    grado_muchos = fuzz.interp_membership(x, pixeles_muchos, num_pixeles)

    # Imprimir los grados de pertenencia
    print(f"Número de píxeles: {num_pixeles}")
    print(f"Grado de pertenencia a pocos píxeles: {grado_pocos}")
    print(f"Grado de pertenencia a muchos píxeles: {grado_muchos}")

    # Aplicar una regla simple de lógica difusa
    if grado_muchos > grado_pocos:
        return "Es un objeto"
    else:
        return "No es un objeto"

if __name__ == "__main__":
    # Capturar imagen de la cámara (reemplazar con tu propia lógica para obtener imágenes)
    image_path = json.load(open("config.json"))["img_path"] + "barco.jpg"
    imagen_camara = cv2.imread(image_path)
    scale_percent = 40  # percent of original size
    width = int(imagen_camara.shape[1] * scale_percent / 100)
    height = int(imagen_camara.shape[0] * scale_percent / 100)
    dim = (width, height)
    imagen_camara = cv2.resize(imagen_camara, dim, interpolation=cv2.INTER_AREA)

    # Definir el rango de color (ajustar según el objeto que estás buscando)
    rango_color = [[0, 100, 0], [100, 255, 200]]  # Rango bajo, Rango alto

    # Detectar color y obtener la máscara
    resultado, mascara = detectar_color(imagen_camara, rango_color)

    # Calcular el centroide de la máscara
    centroide = calcular_centroide(mascara)
    print("Centroide:", centroide)

    # Aplicar lógica difusa para determinar la posición
    posicion = aplicar_logica_difusa_posicion(centroide, (width, height), mostrar_grafica=True)
    print("Posición:", posicion)

    # Calcular el número de píxeles del color detectado
    num_pixeles = cv2.countNonZero(mascara)
    print("Número de píxeles:", num_pixeles)

    # Aplicar lógica difusa para determinar si es un objeto
    es_objeto = aplicar_logica_difusa_pixeles(num_pixeles, mostrar_grafica=True)
    print("¿Es un objeto?:", es_objeto)
