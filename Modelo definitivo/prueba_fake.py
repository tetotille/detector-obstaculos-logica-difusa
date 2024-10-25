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

# cp_contorno_difuso.py
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
print("lista completa", lista_completa)


def hay_semejanzas(lista1, lista2, lista3, lista_completa):
    # Comparar cada lista con las demás si no están vacías
    print("Lista 1:", lista1)
    print("Lista 2:", lista2)
    print("Lista 3:", lista3)
    for i in range(len(lista1)):
        x1, y1, _, _ = lista1[i]  # Extraer las coordenadas x1 e y1 del elemento i
        
        # Comparar con lista2 si no está vacía
        if lista2:
            for j in range(len(lista2)):
                x2, y2, _, _ = lista2[j]  # Extraer las coordenadas x2 e y2 del elemento j
                print(f"Comparando elemento {i} de lista1 ({x1}, {y1}) con elemento {j} de lista2 ({x2}, {y2})")

                # Comparar con lista3
                for k in range(len(lista3)):
                    x3, y3, _, _ = lista3[k]  # Extraer las coordenadas x3 e y3 del elemento k
                    print(f"Comparando elemento {i} de lista1 ({x1}, {y1}) con elemento {k} de lista3 ({x3}, {y3})")

                    # Verificar si hay coincidencia
                    if (cp.array_equal(x1, x2) and cp.array_equal(y1, y2)):
                        print("Coincidencia encontrada entre lista1 y lista2")
                        grupo_mayor = modelo_difuso.agrupar_cuadrados(lista_completa)
                        print(grupo_mayor)
                        return
                    elif (cp.array_equal(y1, y3) and cp.array_equal(x1, x3)):
                        print("Coincidencia encontrada entre lista1 y lista3")
                        grupo_mayor = modelo_difuso.agrupar_cuadrados(lista_completa)
                        print(grupo_mayor)
                        return

        # Comparar con lista3 si lista2 está vacía
        if lista3:
            for k in range(len(lista3)):
                x3, y3, _, _ = lista3[k]  # Extraer las coordenadas x3 e y3 del elemento k
                print(f"Comparando elemento {i} de lista1 ({x1}, {y1}) con elemento {k} de lista3 ({x3}, {y3})")

                if (cp.array_equal(y1, y3) and cp.array_equal(x1, x3)):
                    print("Coincidencia encontrada entre lista1 y lista3")
                    grupo_mayor = modelo_difuso.agrupar_cuadrados(lista_completa)
                    print("el objeto es", grupo_mayor)
                    return

                            # Comparar con lista3 si lista2 está vací

        return False  # Retorna False si no se encuentran coincidencias


resultado = hay_semejanzas(lista1, lista3, lista4, lista_completa)
print("Se encontraron coincidencias:", resultado)