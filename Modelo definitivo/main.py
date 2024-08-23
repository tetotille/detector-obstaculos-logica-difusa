import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath
import filters
import utils
import cv2
import threading

def resize_image_bgr(image, new_shape):
    # Obtén las dimensiones originales y las nuevas dimensiones
    orig_height, orig_width, channels = image.shape
    new_height, new_width = new_shape

    # Factor de escala
    scale_y = orig_height / new_height
    scale_x = orig_width / new_width

    # Crea matrices para las nuevas coordenadas
    y = cp.arange(new_height) * scale_y
    x = cp.arange(new_width) * scale_x
    x_grid, y_grid = cp.meshgrid(x, y)

    # Coordenadas de los píxeles vecinos
    x0 = cp.floor(x_grid).astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, orig_width - 1)
    y0 = cp.floor(y_grid).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, orig_height - 1)

    # Coeficientes de interpolación
    x_weight = x_grid - x0
    y_weight = y_grid - y0

    # Inicializar la imagen redimensionada
    resized_image = cp.zeros((new_height, new_width, channels), dtype=image.dtype)

    for c in range(channels):
        Ia = image[y0, x0, c]
        Ib = image[y1, x0, c]
        Ic = image[y0, x1, c]
        Id = image[y1, x1, c]

        # Interpolación bilineal
        resized_image[:, :, c] = (
            Ia * (1 - x_weight) * (1 - y_weight) +
            Ib * (1 - x_weight) * y_weight +
            Ic * x_weight * (1 - y_weight) +
            Id * x_weight * y_weight
        )

    # Asegurarse de que los valores estén dentro del rango [0, 255]
    resized_image = cp.clip(resized_image, 0, 255)

    return resized_image.astype(cp.uint8)

filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
print(filename)  # Esto te permitirá verificar la ruta completa
image = cv2.imread(filename)
# Normalizar la imagen a un rango de 0 a 255
# Convertir la imagen a un arreglo de CuPy
image_cupy = cp.array(image)

# Obtener las dimensiones originales
orig_height, orig_width, channels = image.shape

# Definir el nuevo ancho y alto manteniendo la proporción
new_width = 200
new_height = int(orig_height * new_width / orig_width)

# Redimensionar la imagen
resized_image = resize_image_bgr(image_cupy, (new_height, new_width))


# Cargar y mostrar la imagen original

cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Contenedores para los resultados
edge_result = []
cmeans_3_result = []
cmeans_4_result = []
filtro_h_result = []
filtro_s_result = []

# Definir las funciones que capturan resultados en contenedores
def process_edge(resized_image, result_container1):
    result = cp_contorno_difuso.process_image(resized_image)
    result_container1.append(result)

def process_cmeans_3(resized_image, centroides, result_container2):
    result = cmeeans_3.main(resized_image, centroides)
    result_container2.append(result)

def process_cmeans_4(resized_image, centroides, result_container3):
    result = cmeeans_3.main(resized_image, centroides)
    result_container3.append(result)

def process_filter_h(resized_image, result_container4):
    result = filters.filter_h(resized_image)
    result_container4.append(result)

def process_filter_s(resized_image, result_container5):
    result = filters.filter_s(resized_image)
    result_container5.append(result)

# Crear hilos para los scripts
edge = threading.Thread(target=process_edge, args=(resized_image, edge_result))
centroides_3 = 3
cmeans_3 = threading.Thread(target=process_cmeans_3, args=(resized_image, centroides_3, cmeans_3_result))
centroides_4 = 4
cmeans_4 = threading.Thread(target=process_cmeans_4, args=(resized_image, centroides_4, cmeans_4_result))
filtro_h = threading.Thread(target=process_filter_h, args=(resized_image, filtro_h_result))
filtro_s = threading.Thread(target=process_filter_s, args=(resized_image, filtro_s_result))

# Iniciar los hilos
edge.start()
cmeans_3.start()
cmeans_4.start()
filtro_h.start()
filtro_s.start()

# Esperar a que los hilos terminen
edge.join()

# Acceder y mostrar los resultados después de que los hilos terminen
print("Resultado del proceso de contorno difuso:")
print(edge_result[0])  # Imprime o procesa el resultado

print("Resultado del cmeans con 3 centroides:")
print(cmeans_3_result[0])  # Imprime o procesa el resultado

print("Resultado del cmeans con 4 centroides:")
print(cmeans_4_result[0])  # Imprime o procesa el resultado

print("Resultado del filtro H:")
print(filtro_h_result[0])  # Imprime o procesa el resultado

print("Resultado del filtro S:")
print(filtro_s_result[0])  # Imprime o procesa el resultado
