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

filename = join(dirname(dirname(abspath(__file__))), "img/brillo.jpg")
image = cv2.imread(filename)
image_cupy = cp.array(image)
lista_completa = []

# Redimensionar la imagen
new_width = 200
orig_height, orig_width, channels = image_cupy.shape
new_height = int(orig_height * new_width / orig_width)
resized_image = utils.resize_image_bgr(image_cupy, (new_height, new_width))

edge_result = cp_contorno_difuso.process_image(resized_image)

mask_max_cluster, fila_interes = cmeeans_3.fcm2(resized_image, 3) 
fila_interes=50
combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
lista1, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista1)

mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 4)  
fila_interes=50
combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
lista2, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista2)

h, fila_interes = filters.filter_h(resized_image)
fila_interes=50
combined_image, contour_image = utils.hacer_mascara(h, fila_interes)
lista3, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista3)

h, fila_interes = filters.filter_s(resized_image)
fila_interes=50
combined_image, contour_image = utils.hacer_mascara(h, fila_interes)
lista4, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
lista_completa.extend(lista4)
print("lista completa", lista_completa)

def hay_semejanzas(lista1, lista2, lista3, lista_completa):
    # Recorrer lista1 y comparar sus elementos con las otras listas
    for i in range(len(lista1)):
        x1, y1, _, _ = lista1[i]  # Extraer las coordenadas x1 e y1 del elemento i
        
        # Buscar coincidencia en lista2
        for j in range(len(lista2)):
            x2, y2, _, _ = lista2[j]
            if x1 == x2 and y1 == y2:  # Coincidencia entre lista1 y lista2
                
                # Si coincide en lista2, buscar coincidencia en lista3
                for k in range(len(lista3)):
                    x3, y3, _, _ = lista3[k]
                    if x1 == x3 and y1 == y3:  # Coincidencia en las tres listas
                        print(f"Coincidencia encontrada en lista1, lista2 y lista3 en el cuadrado ({x1}, {y1})")
                        grupo_mayor = modelo_difuso.agrupar_cuadrados(lista_completa)
                        print("el objeto es", grupo_mayor)
                        return True
    
    # Si no se encuentra ninguna coincidencia en las tres listas
    print("No se encontró ningún cuadrado presente en las tres listas.")
    return False


resultado = hay_semejanzas(lista2, lista3, lista4, lista_completa)
print("Se encontraron coincidencias:", resultado)