import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath
import cv2
import time
import utils
import threading 
import modelo_difuso
import filters


def procesar_imagen(filename):

    image = cv2.imread(filename)
    image_cupy = cp.array(image)
    # Redimensionar la imagen
    new_width = 200
    orig_height, orig_width, channels = image_cupy.shape
    new_height = int(orig_height * new_width / orig_width)
    resized_image = utils.resize_image_bgr(image_cupy, (new_height, new_width))

    edge_result = cp_contorno_difuso.process_image(resized_image)
    lista_completa = []
    # Aplicar el algoritmo FCM con diferentes configuraciones
    mask_max_cluster, fila_interes = cmeeans_3.fcm2(resized_image, 3)
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

def hay_semejanzas(lista1, lista2, lista3, lista_completa):
    coincidencias = []
    
    for i in range(len(lista1)):
        x1, y1, _, _ = lista1[i]
        
        # Buscar coincidencia en lista2
        for j in range(len(lista2)):
            x2, y2, _, _ = lista2[j]
            if cp.array_equal(x1, x2) and cp.array_equal(y1, y2):  # Coincidencia entre lista1 y lista2
                
                # Buscar coincidencia en lista3
                for k in range(len(lista3)):
                    x3, y3, _, _ = lista3[k]

                    if cp.array_equal(x1, x3) and cp.array_equal(y1, y3):  # Coincidencia en las tres listas
                        coincidencia = (x1, y1)
                        coincidencias.append(coincidencia)
    if coincidencias:
        grupo_mayor = modelo_difuso.agrupar_cuadrados(lista_completa)
        return grupo_mayor
    
    return coincidencias, None
