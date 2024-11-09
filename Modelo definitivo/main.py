import cupy as cp
import cmeeans_3
import cp_contorno_difuso
from os.path import join, dirname, abspath, exists
import cv2
import time
import utils
import threading 
import modelo_difuso2
import filters
import detectar_horizonte2
# Configuración de la imagen
filename = join(dirname(dirname(abspath(__file__))), "img/brillo.jpg")
image = cv2.imread(filename)
image_cupy = cp.array(image)
lista_completa = []

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

mascara=0
fila_interes=0
# Contenedores para almacenar los tiempos
task_times = []

# Ejecutar la tarea lenta en su propio stream
def slow_task():
    global slow_task_done
    start_time = time.time()
    print(f"Tarea lenta iniciada a {start_time:.6f}")
    
    with stream_slow:
        edge_result = cp_contorno_difuso.process_image(resized_image)
        fila_interes, imagen = detectar_horizonte2.find_horizontal_line(resized_image)
        _, mascara = utils.crop_horizontal(edge_result, fila_interes)

    
    stream_slow.synchronize()
    end_time = time.time()
    print(f"Tarea lenta finalizada a {end_time:.6f}, duración: {end_time - start_time:.6f} segundos")
    slow_task_done = True  # Marca la tarea lenta como terminada
    task_times.append(("Tarea lenta", start_time, end_time))

# Ejecutar las tareas rápidas
def fast_task_1():
    start_time = time.time()
    print(f"Tarea rápida 1 iniciada a {start_time:.6f}")
    
    with stream_fast_1:
        mask_max_cluster, fila_interes = cmeeans_3.fcm2(resized_image, 3, fila_interes) 
        combined_image, contour_image = utils.hacer_mascara2(mask_max_cluster, mascara)
        lista1, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
        lista_completa.extend(lista1)
    
    stream_fast_1.synchronize()
    end_time = time.time()
    print(f"Tarea rápida 1 finalizada a {end_time:.6f}, duración: {end_time - start_time:.6f} segundos")
    task_times.append(("Tarea rápida 1", start_time, end_time))

def fast_task_2():
    start_time = time.time()
    print(f"Tarea rápida 2 iniciada a {start_time:.6f}")
    
    with stream_fast_2:
        mask_max_cluster, fila_interes = cmeeans_3.fcm(resized_image, 4, fila_interes)  
        combined_image, contour_image = utils.hacer_mascara2(mask_max_cluster, mascara)
        lista2, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
        lista_completa.extend(lista2)
    
    stream_fast_2.synchronize()
    end_time = time.time()
    print(f"Tarea rápida 2 finalizada a {end_time:.6f}, duración: {end_time - start_time:.6f} segundos")
    task_times.append(("Tarea rápida 2", start_time, end_time))
    

def fast_task_3():
    start_time = time.time()
    print(f"Tarea rápida 3 iniciada a {start_time:.6f}")
    
    with stream_fast_3:
        h, fila_interes = filters.filter_h(resized_image, fila_interes)
        combined_image, contour_image = utils.hacer_mascara2(h, mascara)
        lista3, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
        lista_completa.extend(lista3)
    
    stream_fast_3.synchronize()
    end_time = time.time()
    print(f"Tarea rápida 3 finalizada a {end_time:.6f}, duración: {end_time - start_time:.6f} segundos")
    task_times.append(("Tarea rápida 3", start_time, end_time))

def fast_task_4():
    start_time = time.time()
    print(f"Tarea rápida 4 iniciada a {start_time:.6f}")
    
    with stream_fast_4:
        
        h, fila_interes = filters.filter_s(resized_image, fila_interes)
        combined_image, contour_image = utils.hacer_mascara2(h, mascara)
        lista4, constante = utils.segment_and_identify_objects(combined_image, contour_image, combined_image)
        lista_completa.extend(lista4)
        constante = 15
        print("lista",lista_completa)
 
    
    stream_fast_4.synchronize()
    end_time = time.time()
    print(f"Tarea rápida 4 finalizada a {end_time:.6f}, duración: {end_time - start_time:.6f} segundos")
    task_times.append(("Tarea rápida 4", start_time, end_time))

# Ejecutar las tareas rápidas y lenta en paralelo
thread_slow = threading.Thread(target=slow_task)
thread_fast_1 = threading.Thread(target=fast_task_1)
thread_fast_2 = threading.Thread(target=fast_task_2)
thread_fast_3 = threading.Thread(target=fast_task_3)
thread_fast_4 = threading.Thread(target=fast_task_4)

thread_slow.start()
thread_fast_1.start()
thread_fast_2.start()
thread_fast_3.start()
thread_fast_4.start()

thread_slow.join()
thread_fast_1.join()
thread_fast_2.join()
thread_fast_3.join()
thread_fast_4.join()

print("Resumen de tiempos:")
for task, start, end in task_times:
    print(f"{task}: {start:.6f} - {end:.6f}, duración: {end - start:.6f} segundos")
    constante = 15
    print(lista_completa)
    resultado = modelo_difuso2.agrupar_cuadrados(lista_completa, constante)



