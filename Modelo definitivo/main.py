import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath
import filters
import cv2
import time
import utils 

# Configuración de la imagen
filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
image = cv2.imread(filename)
image_cupy = cp.array(image)

# Redimensionar la imagen
new_width = 200
orig_height, orig_width, channels = image_cupy.shape
new_height = int(orig_height * new_width / orig_width)
resized_image = utils.resize_image_bgr(image_cupy, (new_height, new_width))

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
# Crear listas para almacenar los resultados
fast_results_1 = []
fast_results_2 = []
fast_results_3 = []
fast_results_4 = []

while True:
    # Sincronización para la tarea lenta
    stream_slow.synchronize()  # Si la tarea lenta ha terminado, la sincronización completa sin problemas

    # Si la tarea lenta ha terminado, salir del bucle
    if slow_task_done:
        break

    # Ejecución de tareas rápidas
    start_fast_1 = time.time()
    with stream_fast_1:
        mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 3)  
        combined_image, contour_image = utils.hacer_mascara(mask_max_cluster, fila_interes)
        fast_result_1= utils.segment_and_identify_objects(combined_image, contour_image, combined_image)

    stream_fast_1.synchronize()
    end_fast_1 = time.time()
    fast_results_1.append(fast_result_1)  # Guardar el resultado en la lista
    print(f"Fast task 1 (cmeans 3 centroides) took {end_fast_1 - start_fast_1:.6f} seconds")

    start_fast_2 = time.time()
    with stream_fast_2:
        fast_result_2 = cmeeans_3.main(resized_image, 4)
    stream_fast_2.synchronize()
    end_fast_2 = time.time()
    fast_results_2.append(fast_result_2)  # Guardar el resultado en la lista
    print(f"Fast task 2 (cmeans 4 centroides) took {end_fast_2 - start_fast_2:.6f} seconds")

    start_fast_3 = time.time()
    with stream_fast_3:
        fast_result_3 = filters.filter_h(resized_image)
    stream_fast_3.synchronize()
    end_fast_3 = time.time()
    fast_results_3.append(fast_result_3)  # Guardar el resultado en la lista
    print(f"Fast task 3 (filter H) took {end_fast_3 - start_fast_3:.6f} seconds")

    start_fast_4 = time.time()
    with stream_fast_4:
        fast_result_4 = filters.filter_s(resized_image)
    stream_fast_4.synchronize()
    end_fast_4 = time.time()
    fast_results_4.append(fast_result_4)  # Guardar el resultado en la lista
    print(f"Fast task 4 (filter S) took {end_fast_4 - start_fast_4:.6f} seconds")

    # Añadir un retardo opcional entre iteraciones
    time.sleep(0.1)

    # Verificar si la tarea lenta ha terminado después de ejecutar las tareas rápidas
    try:
        stream_slow.synchronize()
        slow_task_done = True  # Marca la tarea lenta como terminada
    except cp.cuda.runtime.CUDARuntimeError:
        slow_task_done = False  # La tarea lenta aún se está ejecutando

# Imprimir o procesar los resultados después de que el bucle termina
print("Resultados de las tareas rápidas:")
print(f"cmeans 3 centroides: {fast_results_1}")
print(f"cmeans 4 centroides: {fast_results_2}")
print(f"filter H: {fast_results_3}")
print(f"filter S: {fast_results_4}")


