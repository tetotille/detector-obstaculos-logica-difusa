import cmeans_torch
import torch_contorno_difuso  # Asumiendo que también migrarás esta librería si usa CuPy internamente
from os.path import join, dirname, abspath, exists
import cv2
import utils_torch  # Asegúrate de que utils también esté adaptado para trabajar con PyTorch
import modelo_difuso_torch
import filters_torch
import detectar_horizonte_torch
import torch

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
    image_tensor = torch.tensor(image).cuda()  # Convertir imagen a tensor de PyTorch y mover a GPU
    
    # Redimensionar la imagen
    new_width = 200
    orig_height, orig_width, channels = image_tensor.shape
    new_height = int(orig_height * new_width / orig_width)
    resized_image = utils_torch.resize_image_bgr(image_tensor, (new_height, new_width))
    
    edge_result = torch_contorno_difuso.process_image(resized_image) 
    fila_interes, imagen = detectar_horizonte_torch.find_horizontal_line(resized_image)
    _, mascara = utils_torch.crop_horizontal(edge_result, fila_interes)
    
    lista_completa = [] 
    # Comparar fila_interes con el valor anterior (si existe)
    if fila_interes_anterior is not None and fila_interes > fila_interes_anterior + 10:
        return "Hay un objeto que cubre el horizonte..."  # Detener ejecución si la condición se cumple
    
    # Guardar el valor actual de fila_interes para la próxima ejecución
    guardar_fila_interes(fila_interes)

    mask_max_cluster, fila_interes = cmeans_torch.fcm2(resized_image, 3, fila_interes)
    # Continuar procesamiento si no se cumple la condición anterior
    combined_image, contour_image = utils_torch.hacer_mascara2(mask_max_cluster, mascara)
    lista1, constante = utils_torch.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista1)
    print(lista1)

    mask_max_cluster, fila_interes = cmeans_torch.fcm(resized_image, 4, fila_interes)
    combined_image, contour_image = utils_torch.hacer_mascara2(mask_max_cluster, mascara)
    lista2, constante = utils_torch.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista2)
    print(lista2)

    # Aplicar filtros adicionales
    h, fila_interes = filters_torch.filter_h(resized_image, fila_interes)
    combined_image, contour_image = utils_torch.hacer_mascara2(mask_max_cluster, mascara)
    lista3, constante = utils_torch.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista3)
    print(lista3)

    h, fila_interes = filters_torch.filter_s(resized_image, fila_interes)
    combined_image, contour_image = utils_torch.hacer_mascara2(mask_max_cluster, mascara)
    lista4, constante = utils_torch.segment_and_identify_objects(combined_image, contour_image, combined_image)
    lista_completa.extend(lista4)
    print(lista4)

    #print(torch.cuda.device_count())  # Usar torch para obtener el número de dispositivos GPU

    # Llamar a la función hay_semejanzas
    grupo_mayor = hay_semejanzas(lista2, lista3, lista4, lista_completa)
    
    return grupo_mayor


def hay_semejanzas(lista1, lista2, lista3, lista_completa):
    coincidencias = []

    # Comparar los elementos de lista1 con lista2 y lista3
    for (x1, y1, _, _) in lista1:
        for (x2, y2, _, _) in lista2:
            if torch.equal(x1, x2) and torch.equal(y1, y2):  # Coincidencia entre lista1 y lista2
                for (x3, y3, _, _) in lista3:
                    if torch.equal(x1, x3) and torch.equal(y1, y3):  # Coincidencia en las tres listas
                        coincidencias.append((x1, y1))  # Guardar coincidencia como tupla (tensor, tensor)

    if coincidencias:
        # Si hay coincidencias, agrupar los cuadrados
        grupo_mayor = modelo_difuso_torch.agrupar_cuadrados(lista_completa)
        return grupo_mayor

    return coincidencias, None

coordenadas = procesar_imagen(r"C:\Users\Koki\detector-obstaculos-logica-difusa\detector-obstaculos-logica-difusa\img\barco.jpg")
