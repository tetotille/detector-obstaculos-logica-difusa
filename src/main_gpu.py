import sys
import os
try:
    import cupy as cp
except:
    import numpy as cp

# Añadir el directorio raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import cv2
from os.path import dirname, abspath,join
from src.cmeans import fcm
from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv import detector_hsv,detector_rgb_gpu
from src.utils import read_image,FrameMemory
import serial
import struct
import time

from src.fuzzy_union.fuzzy_union import fuzzy_union


    # Configura el puerto serial donde está conectado tu sensor LiDAR
try:
    ser = serial.Serial(
        port='COM3',  # Cambia esto al puerto UART de tu TX2
        baudrate=115200,
        timeout=1
        )
except:
    print("Lidar no detectado")

image_folder = join(dirname(dirname(abspath(__file__))),"assets/images")
image_set = {f"{join(image_folder,'akaso1.jpeg')}",
             f"{join(image_folder,'akaso2.jpeg')}",
             f"{join(image_folder,'ypacarai.jpeg')}",
             f"{join(image_folder,'barco.jpg')}",
             f"{join(image_folder,'lago-ypacarai (6).jpg')}",
             f"{join(image_folder,'IMG_6830.jpeg')}",
             f"{join(image_folder,'akaso3.jpeg')}",
             }

def send_command(command):
    """Envía un comando al sensor."""
    ser.write((command + '\n').encode('ascii'))

def read_data_block():
    """Lee un bloque de datos del sensor (7 bytes)."""
    data = ser.read(7)
    if len(data) == 7:
        # Asumiendo que los datos siguen el formato: <BHHBB>
        sync, azimuth, distance, strength, checksum = struct.unpack('<BHHBB', data)
        return {
            'sync': sync,
            'azimuth': azimuth / 100.0,  # Conversión a grados
            'distance': distance,          # Distancia en cm
            'strength': strength,          # Intensidad de la señal
            'checksum': checksum
        }
    else:
        return None


def get_video_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yield frame

    cap.release()

def get_image_frame():
    image_path = image_set.pop() 
    # print(f"Image name: {image_path.split('/')[-1]}")
    image = cv2.imread(image_path)
    filename = image_path.split('/')[-1]
    return image,filename

def main():
    # Cuando se use el video
    # while True:
        # frame = get_video_stream()

    # Cuando se use una imagen
    import pickle
    i = 0
    counter = 0
    tics = {}
    repetir = 100
    total = repetir * len(image_set)
    while image_set:
        i += 1
        frame,filename = get_image_frame()
        tics[filename] = {"total":[],"horizon":[],"rgb":[],"cmeans":[]}
        for _ in range(repetir):
            counter += 1
            init = time()
            image_np,image_cp = read_image(frame,256,192)
            left,center,right = separate_pixels(image_np)
            
            left = find_largest_fuzzy_jump(left)
            center = find_largest_fuzzy_jump(center)
            right = find_largest_fuzzy_jump(right)

            tic1 = time()
            if abs(center - left) < abs(right - center) and abs(center - left) < abs(right - left):
                a,b = left,center
            elif abs(center - left) > abs(right - center) and abs(right - center) < abs(right - left):
                a,b = center,right
            else:
                a,b = left,right
            
            # Draw horizontal lines at the fuzzy jump indices
            #cv2.line(image_np, (0, a), (image_np.shape[1], b), (0, 255, 0), 2)
            cropped_image_np = image_np[(a+b)//2:,:]
            cropped_image_cp = image_cp[(a+b)//2:,:]

            ajuste = (a+b)//2
            tic2 = time()
            hsv_cp, cuadros_rgb = detector_rgb_gpu(cropped_image_cp)
            tic3 = time()
            cmeans_image, cmeans_fila_interes, cuadros_cmeans = fcm(image_cp,4,punto_horizonte=ajuste-20)
            tic4 = time()
            encuadrar = True
            for cuadro in cuadros_rgb:
                cuadro["y_centroid"] = cuadro["y_centroid"] + ajuste - (a+b)//2
            cons = 0
            for k in range(len(cuadros_cmeans)):
                j = k + cons
                cuadros_cmeans[j]["y_centroid"] = cuadros_cmeans[j]["y_centroid"] + cmeans_fila_interes - (a+b)//2
                if cuadros_cmeans[j]["y_centroid"] < 0:
                    del cuadros_cmeans[j]
                    cons -= 1
            if encuadrar:
                for cuadro in cuadros_rgb:
                    cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]+ajuste), (cuadro["x_end"],cuadro["y_end"]+ajuste), (0, 0, 255), 2)
                for cuadro in cuadros_cmeans:
                    cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]+cmeans_fila_interes), (cuadro["x_end"],cuadro["y_end"]+cmeans_fila_interes), (0, 255, 0), 2)

                cv2.imwrite("main_output/" + filename.split(".")[0] + ".png", image_np)
                try:
                    cv2.imwrite("main_output/rgb_" + filename.split(".")[0] + ".png", hsv_cp.get())
                    cv2.imwrite("main_output/cmeans_" + filename.split(".")[0] + ".png", cmeans_image.get())
                except:
                    cv2.imwrite("main_output/rgb_" + filename.split(".")[0] + ".png", hsv_cp)
                    cv2.imwrite("main_output/cmeans_" + filename.split(".")[0] + ".png", cmeans_image)
            
            tics[filename]["total"].append(time()-init)
            tics[filename]["horizon"].append(tic1-init)
            tics[filename]["rgb"].append(tic3-tic2)
            tics[filename]["cmeans"].append(tic4-tic3)
            
            
            fuzzy_frames = fuzzy_union([cuadros_rgb, cuadros_cmeans])
            """Explicación de fuzzy_union
            fuzzy_frames:list[dict] = [
                {
                    "puntos": [...],        # lista de puntos que compete al obstáculo
                    "x_init": int,          # límite izquierdo del obstáculo
                    "x_end": int,           # límite derecho del obstáculo
                    "y_init": int,          # límite superior del obstáculo
                    "y_end": int,           # límite inferior del obstáculo
                    "x": int,               # sumatoria de todos los valores de x de cada punto
                    "y": int,               # sumatoria de todos los valores de y de cada punto
                    "weight": int,          # peso del obstáculo
                    "x_centroid": int,      # centro de masa del obstáculo en el eje x
                    "y_centroid": int,      # centro de masa del obstáculo en el eje y
                    "distancia_minima": int,# distancia mínima al obstáculo detectado en otro detector
                    "fuzzy_union": int      # puntaje de 0 a 1 de la unión de los detectores
                },
                {...},
                ...
            ]
            
            
            """
            tic_final = time()
            # print("RGB Detector")
            # print(cuadros_rgb)g
            # print("\n\nCmeans Detector")
            # print(cuadros_cmeans)
            # print(fuzzy_frames)
            # print(f"Tiempo total: {time()-init}")
            print(f"Ha terminado el {(counter*100)//total}%",end="\r")
    pickle.dump(tics,open("main_output/tics.pkl","wb"))
    try:
        send_command('DS')
    except:
        print("No hay sensor")
    
def main_video():
    video_memory:list[FrameMemory] = []
    memory_limit = 5 # cantidad de frames de memoria
    max_x = 256
    max_y = 192
    while True:
        frame = get_video_stream()
        image_np,frame_gpu = read_image(frame,max_x,max_y)
        left,center,right = separate_pixels(frame_gpu)
            
        left = find_largest_fuzzy_jump(left)
        center = find_largest_fuzzy_jump(center)
        right = find_largest_fuzzy_jump(right)

        tic1 = time()
        if abs(center - left) < abs(right - center) and abs(center - left) < abs(right - left):
            a,b = left,center
        elif abs(center - left) > abs(right - center) and abs(right - center) < abs(right - left):
            a,b = center,right
        else:
            a,b = left,right
        
        cropped_image_cp = frame_gpu[(a+b)//2:,:]
        
        ajuste = (a+b)//2
        
        ##### DETECTOR 1 ######
        hsv_cp, cuadros_rgb = detector_rgb_gpu(cropped_image_cp)
        
        ##### DETECTOR 2 ######
        cmeans_image, cmeans_fila_interes, cuadros_cmeans = fcm(frame_gpu,4,punto_horizonte=ajuste-20)
        
        ###### DETECTOR LIDAR ######
        lidar:tuple[float,float] = (3.3,20.2) # Acá se define la función
        
        for cuadro in cuadros_rgb:
            cuadro["y_centroid"] = cuadro["y_centroid"] + ajuste - (a+b)//2
        
        for i in range(len(cuadros_cmeans)):
            j = j + i
            cuadros_cmeans[j]["y_centroid"] = cuadros_cmeans[j]["y_centroid"] + cmeans_fila_interes - (a+b)//2
            if cuadros_cmeans[j]["y_centroid"] < 0:
                del cuadros_cmeans[j]
                j -= 1
        
        ###### UNION DETECTORES ######
        fuzzy_frames = fuzzy_union([cuadros_rgb, cuadros_cmeans],lidar)
        """Explicación de fuzzy_union
            fuzzy_frames:list[dict] = [
                {
                    "puntos": [...],        # lista de puntos que compete al obstáculo
                    "x_init": int,          # límite izquierdo del obstáculo
                    "x_end": int,           # límite derecho del obstáculo
                    "y_init": int,          # límite superior del obstáculo
                    "y_end": int,           # límite inferior del obstáculo
                    "x": int,               # sumatoria de todos los valores de x de cada punto
                    "y": int,               # sumatoria de todos los valores de y de cada punto
                    "weight": int,          # peso del obstáculo
                    "x_centroid": int,      # centro de masa del obstáculo en el eje x
                    "y_centroid": int,      # centro de masa del obstáculo en el eje y
                    "distancia_minima": int,# distancia mínima al obstáculo detectado en otro detector
                    "fuzzy_union": int      # puntaje de 0 a 1 de la unión de los detectores
                },
                {...},
                ...
            ]
        """
        ############## IMPLEMENTACIÓN DE MEMORIA ################
        """
            Se utiliza fuzzy_union, y generalmente éste tiene pocos valores, 1 o 2 por frame, por lo que un for no
            va a ser prácticamente una carga para la cpu

        """
        # Agregado a la memoria
        for fuzzy_frame in fuzzy_frames:
            in_memory = False
            for memory in video_memory:
                if fuzzy_frame in memory:
                    memory.add(fuzzy_frame)
                    memory.modified= True
                    in_memory = True
            if not in_memory:
                memory = FrameMemory(memory_limit,fuzzy_frame)
                video_memory.append(memory)

        # Limpiado de memoria
        max_weight = 0
        max_score = 0
        final_score = 0
        for memory in video_memory[:]:
            if not memory.modified:
                memory.add(None)
            memory.modified = False
            if memory.empty():
                video_memory.remove(memory)
            max_weight = max(max_weight,memory.get_weight())     # Peso (cantidad de pixeles) del obstáculo
            max_score = max(max_score,memory.score())           # Cantidad de frames en los que se detectó el obstáculo
            if final_score < max_weight * (max_score/2):
                x,y = memory.get_point()         # Coordenadas del centroide del obstáculo

        #########################################################

        ######## UTILIZACIÓN DE RESULTADOS ###########
        """
           Se puede obtener los puntos x e y de cada obstáculo detectado de la siguiente forma:
           suponiendo que se recorre video_memory

           Para este caso solo se tiene en cuenta el obstáculo con el score más alto, es decir el que tenga el final_score más alto
           x,y: son el centroide del obstáculo que debe ser tomado mayormente en cuenta
        """
        # Resultado te dice si el objeto está a la izquierda, derecha o centro
        resultado = "Izquierda" if x < max_x//2 else "Derecha" if x > max_x//2 else "Centro"
        ##############################################

        data_block = read_data_block()
        if data_block:
            azimuth = data_block['azimuth']
            distance = data_block['distance']
            strength = data_block['strength']

            # Filtrar los datos por el rango de ángulo -90 a 90 grados
            if -90 <= azimuth <= 90 and distance < 40 :
                lidar=(f"Ángulo: {azimuth:.2f}°, Distancia: {distance} cm, Intensidad: {strength}")
            else:
                print("No se pudo leer un bloque de datos. Verifique la conexión.")
            

if __name__ == "__main__":

    from time import time

    # HAY QUE CAMBIAR EL MAIN POR main_video() PARA LAS PRUEBAS FINALES
    main()


