import skfuzzy as fuzz
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, abspath, join
from sys import argv
from utils import crop_horizontal
from src.detector_horizonte.horizonte_mar_rojo_cielo_azul import detectar_horizonte

def filter_h(img):
     #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h,_,_ = cv2.split(img_hsv)
    
    h = 255 - h

    num_bins = 256
    fuzzy_hist = np.zeros(num_bins)

    x_intensities = np.arange(0, 256, 1)
    
    for i in range(num_bins):
        # Crear la función de pertenencia trapezoidal para el intervalo
        if i == 0:
            # Trapezoide inicial
            mf = fuzz.trapmf(x_intensities, [0, 0, 1, 2])

        elif i == num_bins - 1:
            # Trapezoide final
            mf = fuzz.trapmf(x_intensities, [254, 255, 255, 255])

        else:
            # Trapezoide intermedio
            mf = fuzz.trapmf(x_intensities, [i-1, i, i+1, i+2])
        
        # Calcular el grado de pertenencia de cada píxel a este intervalo
        membership_values = fuzz.interp_membership(x_intensities, mf, h)
        
        # Sumar los valores de pertenencia para formar el histograma difuso
        fuzzy_hist[i] = np.sum(membership_values)
    
    max_index = np.argmax(fuzzy_hist)
    most_frequent_intensity = x_intensities[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    h_filtrada = np.copy(h)
    h_filtrada[(h >= most_frequent_intensity-10) & (h <= most_frequent_intensity+10)] = 0
    return h_filtrada

def filter_s(img_path):
    #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)
    
    _,s,_ = cv2.split(img_hsv)
    
    s = 255 - s

    hist, bins = np.histogram(s.ravel(), 256, [0, 256])

    max_index = np.argmax(hist)
    most_frequent_intensity = bins[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    s_filtrada = np.copy(s)
    s_filtrada[(s >= most_frequent_intensity-10) & (s <= most_frequent_intensity+10)] = 0

    return s_filtrada

import cupy as cp

import cupy as cp

def segment_and_identify_objects(image_gray, mask_binary, original, block_size=15, threshold_area=155):
    # Paso 1: Analizar bloques de block_size x block_size píxeles
    height, width, _ = image_gray.shape
    
    # Crear grids de índices para recorrer en bloques
    y_indices, x_indices = cp.meshgrid(cp.arange(block_size, height - block_size, block_size),
                                       cp.arange(block_size, width - block_size, block_size),
                                       indexing='ij')

    # Generar las coordenadas de los bloques de forma compatible con el broadcasting
    y_offsets = cp.arange(block_size).reshape(1, block_size, 1)
    x_offsets = cp.arange(block_size).reshape(block_size, 1, 1)

    # Reestructurar las imágenes para aplicar operaciones en bloques
    block_images = image_gray[y_indices[:, :, None] + y_offsets,
                              x_indices[:, None, :] + x_offsets]

    block_masks = mask_binary[y_indices[:, :, None] + y_offsets,
                              x_indices[:, None, :] + x_offsets]
    
    # Contar los píxeles negros en cada bloque
    black_pixel_counts = cp.sum(cp.all(block_images == cp.array([0, 0, 0]), axis=-1), axis=(2, 3))

    # Detectar píxeles verdes en la máscara
    green_pixel_masks = (block_masks[:, :, :, 1] > 100) & (block_masks[:, :, :, 0] < 50) & (block_masks[:, :, :, 2] < 50)
    green_pixel_counts = cp.sum(green_pixel_masks, axis=(2, 3))

    # Condición para identificar bloques
    valid_blocks = (black_pixel_counts > threshold_area) & (green_pixel_counts <= 40)

    # Procesar bloques adyacentes para verificar si cumplen con las condiciones
    # Crear un padding alrededor de la matriz para manejar bordes
    padded_valid_blocks = cp.pad(valid_blocks, ((1, 1), (1, 1)), mode='constant', constant_values=False)

    # Desplazamientos para los vecinos (arriba, abajo, izquierda, derecha, y diagonales)
    shifts = [
        (0, 1),  # derecha
        (1, 1),  # diagonal abajo derecha
        (-1, 1), # diagonal arriba derecha
        (0, -1), # izquierda
        (1, -1), # diagonal abajo izquierda
        (-1, -1),# diagonal arriba izquierda
        (1, 0),  # abajo
        (-1, 0)  # arriba
    ]
    
    # Inicializar una matriz para almacenar si el bloque tiene vecinos válidos
    adyacente_verificado = cp.zeros_like(valid_blocks, dtype=cp.bool_)

    for dy, dx in shifts:
        # Comparar bloques válidos con sus vecinos desplazados
        vecinos = padded_valid_blocks[1 + dy:height//block_size + 1 + dy, 1 + dx:width//block_size + 1 + dx]
        adyacente_verificado |= vecinos  # Si cualquier vecino es válido, marcarlo

    # Obtener los bloques que son válidos y tienen al menos un vecino válido
    final_valid_blocks = valid_blocks & adyacente_verificado

    # Aplanar los índices válidos
    valid_y_indices, valid_x_indices = cp.where(final_valid_blocks)
    
    # Inicializar rectángulos detectados
    rects = [(x_indices[y, x], y_indices[y, x], block_size, block_size) for y, x in zip(valid_y_indices, valid_x_indices)]

    # Marcar los rectángulos encontrados en la imagen original
    for rect in rects:
        x, y, w, h = rect
        original[y:y+h, x:x+w, :] = cp.array([0, 0, 255])  # Rojo
    
    # Convertir a un formato que pueda mostrar la imagen
    image_to_show = cp.asnumpy(original)

    # Mostrar la imagen con los bloques marcados
    cv2.imshow('Segmented Image with Detected Objects', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usar la función con la ruta de la imagen, la máscara y la imagen original en forma de array de CuPy
if __name__ == "__main__":

    # Ruta a la imagen
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
    img=cv2.imread(filename)
    
    cv2.imshow('imagen final.png',filter_h(img))
    cv2.waitKey(0)
    x,y,a = img.shape
    imag=cv2.resize(img, (200, int(x*200/y)))
    ima, fila_interes = detectar_horizonte(imag)
    imagen=cv2.resize(img, (200, int(x*200/y)))
    _, image3 =crop_horizontal(filter_h(imagen), fila_interes)
    print(fila_interes)

    # Crea una imagen con transparencia (canal alfa)
    mask=cv2.imread("imagen_umbral.png", cv2.IMREAD_GRAYSCALE)
    x,y = mask.shape
    _, mask2=crop_horizontal(mask, fila_interes)
    # Leer la imagen original (en color)
   
    original = cv2.cvtColor(image3, cv2.IMREAD_COLOR)
    
    # Crear una máscara para los píxeles negros
    mask = cv2.inRange(original, np.array([0, 0, 0]), np.array([15, 15, 15]))  # Ajusta el rango según sea necesario

    # Crear imágenes en negro y blanco
    white_image = np.full_like(original, 255)
    black_image = np.zeros_like(original)

    # Aplicar la máscara para obtener la imagen con negros convertidos a blancos y el resto a negro
    original = cv2.bitwise_and(white_image, white_image, mask=mask) + cv2.bitwise_and(black_image, black_image, mask=cv2.bitwise_not(mask))


    x,y,a = original.shape
    height, width, channels = original.shape
    height1, width1 = mask2.shape
    print(height, width, channels,height1, width1 )
    # Verificar las dimensiones de ambas imágenes

    if (height, width) != (height1, width1):
        print("Redimensionando la máscara para que coincida con las dimensiones de la imagen original.")
        mask2 = cv2.resize(mask2, (width, height))
    else:
        print("Las dimensiones de la máscara ya coinciden con las dimensiones de la imagen original.")


    # Asegurarse de que la máscara sea binaria
    _, mask_binary = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)

    # Crear una imagen de contornos verdes (tamaño de la imagen original)
    contour_image = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)

    # Establecer los contornos en verde (BGR) usando la máscara binaria
    contour_image[mask_binary > 0] = [0, 255, 0]  # Verde en BGR

    # Crear una imagen RGBA con el fondo transparente
    rgba_image = np.zeros((original.shape[0], original.shape[1], 4), dtype=np.uint8)

    # Copiar la imagen original al canal RGB de la imagen RGBA
    rgba_image[:, :, :3] = original

    # Crear una máscara para el canal alfa (transparencia)
    transparency_mask = np.zeros((original.shape[0], original.shape[1]), dtype=np.uint8)
    transparency_mask[mask_binary > 0] = 255  # Píxeles de contorno serán completamente opacos

    # Copiar la máscara de transparencia al canal alfa de la imagen RGBA
    rgba_image[:, :, 3] = transparency_mask

    # Aplicar la imagen de contornos al canal alfa de la imagen RGBA
    # Para hacer que los contornos sean visibles, combinamos la imagen original con la imagen de contornos
    combined_image = cv2.addWeighted(original, 1.0, contour_image, 1.0, 0)

    # Crear una imagen RGBA con el fondo transparente y combinar con la imagen de contornos
    rgba_image[:, :, :3] = combined_image


    #cv2.imwrite(filename, rgba_image)

    # Muestra la imagen resultante
    cv2.imshow("Imagen con Contornos Superpuestos", rgba_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convertir la imagen a escala de grises
    image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# Usar la función con la ruta de la imagen y la máscara
    segment_and_identify_objects(original, contour_image)