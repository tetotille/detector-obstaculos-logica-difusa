import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath
import filters
import cv2
import time

def resize_image_bgr(image, new_shape):
    orig_height, orig_width, channels = image.shape
    new_height, new_width = new_shape

    scale_y = orig_height / new_height
    scale_x = orig_width / new_width

    y = cp.arange(new_height) * scale_y
    x = cp.arange(new_width) * scale_x
    x_grid, y_grid = cp.meshgrid(x, y)

    x0 = cp.floor(x_grid).astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, orig_width - 1)
    y0 = cp.floor(y_grid).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, orig_height - 1)

    x_weight = x_grid - x0
    y_weight = y_grid - y0

    resized_image = cp.zeros((new_height, new_width, channels), dtype=image.dtype)

    for c in range(channels):
        Ia = image[y0, x0, c]
        Ib = image[y1, x0, c]
        Ic = image[y0, x1, c]
        Id = image[y1, x1, c]

        resized_image[:, :, c] = (
            Ia * (1 - x_weight) * (1 - y_weight) +
            Ib * (1 - x_weight) * y_weight +
            Ic * x_weight * (1 - y_weight) +
            Id * x_weight * y_weight
        )

    resized_image = cp.clip(resized_image, 0, 255)
    return resized_image.astype(cp.uint8)

# Configuración de la imagen
filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
image = cv2.imread(filename)
image_cupy = cp.array(image)

# Redimensionar la imagen
new_width = 200
orig_height, orig_width, channels = image_cupy.shape
new_height = int(orig_height * new_width / orig_width)
resized_image = resize_image_bgr(image_cupy, (new_height, new_width))

# Crear streams para ejecutar en paralelo
stream_slow = cp.cuda.Stream()
stream_fast_1 = cp.cuda.Stream()
stream_fast_2 = cp.cuda.Stream()
stream_fast_3 = cp.cuda.Stream()
stream_fast_4 = cp.cuda.Stream()

# Contenedores para los resultados
edge_result = cp.empty_like(resized_image)
cmeans_3_result = cp.empty_like(resized_image)
cmeans_4_result = cp.empty_like(resized_image)
filtro_h_result = cp.empty_like(resized_image)
filtro_s_result = cp.empty_like(resized_image)

# Ejecutar la tarea lenta en su propio stream
with stream_slow:
    edge_result = cp_contorno_difuso.process_image(resized_image)

# Inicializar la variable para verificar el tiempo
slow_task_done = False

# Mientras la tarea lenta se ejecuta, repetir las tareas rápidas
fast_results_1 = []
fast_results_2 = []
fast_results_3 = []
fast_results_4 = []

while not slow_task_done:
    # Verificar si la tarea lenta ha terminado
    try:
        stream_slow.synchronize()  # Si la tarea lenta ha terminado, se sincroniza sin problemas
        slow_task_done = True  # La tarea lenta ha terminado
    except cp.cuda.runtime.CUDARuntimeError:
        slow_task_done = False  # La tarea lenta aún se está ejecutando

    # Ejecución de tareas rápidas
    with stream_fast_1:
        centroides_3 = 3
        fast_result_1 = cmeeans_3.main(resized_image, centroides_3)
    fast_results_1.append(fast_result_1)
    stream_fast_1.synchronize()
    print(f"Fast task 1 (cmeans 3 centroides) completed with result: {fast_result_1}")

    with stream_fast_2:
        centroides_4 = 4
        fast_result_2 = cmeeans_3.main(resized_image, centroides_4)
    fast_results_2.append(fast_result_2)
    stream_fast_2.synchronize()
    print(f"Fast task 2 (cmeans 4 centroides) completed with result: {fast_result_2}")

    with stream_fast_3:
        fast_result_3 = filters.filter_h(resized_image)
    fast_results_3.append(fast_result_3)
    stream_fast_3.synchronize()
    print(f"Fast task 3 (filter H) completed with result: {fast_result_3}")

    with stream_fast_4:
        fast_result_4 = filters.filter_s(resized_image)
    fast_results_4.append(fast_result_4)
    stream_fast_4.synchronize()
    print(f"Fast task 4 (filter S) completed with result: {fast_result_4}")

    # Añadir un retardo opcional entre iteraciones
    time.sleep(0.1)

# Mostrar los resultados finales
print("Resultado del proceso de contorno difuso:")
print(edge_result)

print("Resultados de las tareas rápidas:")
print(f"cmeans 3 centroides: {fast_results_1}")
print(f"cmeans 4 centroides: {fast_results_2}")
print(f"filter H: {fast_results_3}")
print(f"filter S: {fast_results_4}")

