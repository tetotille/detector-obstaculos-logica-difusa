import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath, exists
import cv2
import time
import utils
import threading
import modelo_difuso
import filters

# Ruta del archivo para almacenar fila_interes
FILE_PATH = join(dirname(abspath(__file__)), "fila_interes.txt")

def guardar_fila_interes(fila_interes):
    """Guarda el valor de fila_interes en un archivo de texto."""
    with open(FILE_PATH, 'w') as file:
        file.write(str(fila_interes))

def cargar_fila_interes():
    """Carga el valor de fila_interes desde un archivo de texto si existe, de lo contrario devuelve None."""
    if exists(FILE_PATH):
        with open(FILE_PATH, 'r') as file:
            return int(file.read())
    return None

def procesar_imagen(filename):
    # Cargar el valor previo de fila_interes si existe
    fila_interes_anterior = cargar_fila_interes()

    image = cv2.imread(filename)
    image_cupy = cp.array(image)
    
    # Redimensionar la imagen
    new_width = 200
    orig_height, orig_width, channels = image_cupy.shape
    new_height = int(orig_height * new_width / orig_width)
    resized_image = utils.resize_image_bgr(image_cupy, (new_height, new_width))

    lista_completa = []

    # Aplicar el algoritmo FCM con diferentes configuraciones
    mask_max_cluster, fila_interes = cmeeans_3.fcm2(resized_image, 3)
    
    # Comparar fila_interes con el valor anterior (si existe)
    if fila_interes_anterior is not None and fila_interes < fila_interes_anterior / 10:
        print("Hay un objeto que cubre el horizonte.")
        return None  # Detener ejecución si la condición se cumple

    # Guardar el valor actual de fila_interes para la próxima ejecución
    guardar_fila_interes(fila_interes)

    combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
    lista1, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista1)

    mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 4)
    combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
    lista2, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista2)

    # Aplicar filtros adicionales
    h, fila_interes = filters.filter_h(resized_image)
    combined_image, contour_image = utils.hacer_mascara(h, fila_interes)
    lista3, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista3)

    h, fila_interes = filters.filter_s(resized_image)
    combined_image, contour_image = utils.hacer_mascara(h, fila_interes)
    lista4, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista4)

    return lista_completa, lista1, lista2, lista3, lista4
