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

filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
image = cv2.imread(filename)
image_cupy = cp.array(image)
lista_completa = []

# Redimensionar la imagen
new_width = 200
orig_height, orig_width, channels = image_cupy.shape
new_height = int(orig_height * new_width / orig_width)
resized_image = utils.resize_image_bgr(image_cupy, (new_height, new_width))

edge_result = cp_contorno_difuso.process_image(resized_image)

mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 3)  
combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
lista1, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista1)

mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 4)  
combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
lista2, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista2)

h, fila_interes = filters.filter_h(resized_image)
combined_image, contour_image = utils.hacer_mascara(h, fila_interes)
lista3, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista3)

h, fila_interes = filters.filter_s(resized_image)
combined_image, contour_image = utils.hacer_mascara(h, fila_interes)
lista4, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista4)

grupo_mayor = modelo_difuso.verificar_y_devolver_grupo_mayor(lista_completa, constante)

if grupo_mayor:
    print("Grupo mayor encontrado:", grupo_mayor)
else:
    print("No se encontró un grupo mayor sin empates.")
