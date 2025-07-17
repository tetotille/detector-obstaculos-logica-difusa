import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath, exists
import cv2
import utils
import modelo_difuso2
import filters
import detectar_horizonte2

# Ruta del archivo para almacenar fila_interes
FILE_PATH = join(dirname(abspath(__file__)), "fila_interes.txt")

def guardar_fila_interes(fila_interes):
    """Guarda el valor de fila_interes en un archivo de texto."""
    with open(FILE_PATH, 'w') as file:
        file.write(str(fila_interes))

def cargar_fila_interes():
    """Carga el valor de fila_interes desde un archivo de texto si existe, de lo contrario devuelve None."""
    if exists(FILE_PATH):
        try:
            with open(FILE_PATH, 'r') as file:
                return int(file.read())  # Lee el contenido y lo convierte en entero
        except ValueError:
            print("Error: El archivo contiene un valor no numérico.")
            return None
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
    
    edge_result = cp_contorno_difuso.process_image(resized_image) 
    fila_interes, imagen = detectar_horizonte2.find_horizontal_line(resized_image)
    _, mascara = utils.crop_horizontal(edge_result, fila_interes)
    # Recortar la imagen horizontalmente (supongamos que crop_horizontal también trabaja con CuPy)
    """cv2.imshow("original", mascara.get())
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    lista_completa = [] 
    # Comparar fila_interes con el valor anterior (si existe)
    if fila_interes_anterior is not None and fila_interes > fila_interes_anterior+10:
        return "Hay un objeto que cubre el horizonte..."  # Detener ejecución si la condición se cumple
    # Guardar el valor actual de fila_interes para la próxima ejecución
    guardar_fila_interes(fila_interes)

    mask_max_cluster, fila_interes = cmeeans_3.fcm2(resized_image, 3, fila_interes)
    # Continuar procesamiento si no se cumple la condición anterior
    combined_image, contour_image = utils.hacer_mascara2(mask_max_cluster, mascara)
    lista1, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista1)
    print(lista1)

    mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 4, fila_interes)
    combined_image, contour_image = utils.hacer_mascara2(mask_max_cluster, mascara)
    lista2, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista2)
    print(lista2)

    # Aplicar filtros adicionales
    h, fila_interes = filters.filter_h(resized_image, fila_interes)
    combined_image, contour_image = utils.hacer_mascara2(mask_max_cluster, mascara)
    lista3, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista3)
    print(lista3)

    h, fila_interes = filters.filter_s(resized_image, fila_interes)
    combined_image, contour_image = utils.hacer_mascara2(mask_max_cluster, mascara)
    lista4, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista4)
    print(lista4)

    print(cp.cuda.runtime.getDeviceCount()) 
    # Llamar a la función hay_semejanzas
    grupo_mayor = hay_semejanzas(lista2, lista3, lista4, lista_completa)
    
    return grupo_mayor


def hay_semejanzas(lista1, lista2, lista3, lista_completa):
    coincidencias = []

    # Comparar los elementos de lista1 con lista2 y lista3
    for (x1, y1, _, _) in lista1:
        for (x2, y2, _, _) in lista2:
            if cp.array_equal(x1, x2) and cp.array_equal(y1, y2):  # Coincidencia entre lista1 y lista2
                for (x3, y3, _, _) in lista3:
                    if cp.array_equal(x1, x3) and cp.array_equal(y1, y3):  # Coincidencia en las tres listas"""
                        coincidencias.append((x1, y1))  # Guardar coincidencia como tupla (array, array)

    if coincidencias:
        # Si hay coincidencias, agrupar los cuadrados
        grupo_mayor = modelo_difuso2.agrupar_cuadrados(lista_completa)
        return grupo_mayor

    return coincidencias, None

