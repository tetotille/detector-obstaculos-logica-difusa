import sys
import os
#import subprocess
import requests

try:
    import cupy as cp
except:
    import numpy as cp

# Añadir el directorio raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configuración del programa
server_url = 'http://localhost:5000'  # URL del servidor Flask
endpoint = '/obstaculos'               # Ruta del endpoint
TEST = False

import cv2
from os.path import dirname, abspath,join
from src.cmeans import fcm
from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv import detector_hsv,detector_rgb_gpu
from src.utils import read_image,FrameMemory,VideoStream
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
    LIDAR = True
except:
    LIDAR = False
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


def verificar_conexion(server_url, endpoint="/"):
    """Verifica si el servidor Flask está activo."""
    try:
        response = requests.get(server_url + endpoint, timeout=5)
        if response.status_code == 200:
            print("Servidor Flask activo.")
            return True
        else:
            print(f"Servidor respondió con código: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con el servidor Flask: {e}")
        return False

def cambiar_estado_obstaculos(server_url, endpoint, nuevo_estado, reintentos=5, espera=5):
    """
    Cambia el estado de los obstáculos después de verificar que el servidor Flask está activo.
    
    server_url: URL del servidor Flask (por ejemplo, 'http://localhost:5000')
    endpoint: Endpoint para realizar el POST (por ejemplo, '/cambiar_estado')
    nuevo_estado: Estado que se desea establecer.
    reintentos: Número de intentos para conectar al servidor antes de rendirse.
    espera: Tiempo en segundos entre reintentos.
    """
    intentos = 0
    while intentos < reintentos:
        if verificar_conexion(server_url):
            payload = {"estado": nuevo_estado}
            try:
                response = requests.post(server_url + endpoint, json=payload)
                if response.status_code == 200:
                    print("Estado de los obstáculos cambiado con éxito.")
                    return True
                else:
                    print(f"Error al cambiar el estado: {response.status_code} - {response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Error durante el POST: {e}")
                return False
        else:
            print(f"Intento {intentos + 1}/{reintentos} fallido. Reintentando en {espera} segundos...")
            time.sleep(espera)
            intentos += 1
    
    print("No se pudo conectar al servidor Flask después de varios intentos.")
    return False




def get_video_stream(cap):
    print(f"Res: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    while cap.grab():
        
        ret, frame = cap.retrieve()
        if ret:
            break

    return frame


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
    stream = VideoStream("rtsp://192.168.1.1:554/live")
    # cap = cv2.VideoCapture("rtsp://192.168.1.1:554/live",cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        tic1 = time.time()
        ret, frame = stream.read()
        if not ret or frame is None:
            continue
        tic2 = time.time()
        image_np,frame_gpu = read_image(frame,max_x,max_y)
        tic3 = time.time()
        try:
            left,center,right = separate_pixels(frame_gpu.get())
        except:
            left,center,right = separate_pixels(frame_gpu)
        tic4 = time.time()
            
        left = find_largest_fuzzy_jump(left)
        center = find_largest_fuzzy_jump(center)
        right = find_largest_fuzzy_jump(right)
        tic5 = time.time()
            
        if abs(center - left) < abs(right - center) and abs(center - left) < abs(right - left):
            a,b = left,center
        elif abs(center - left) > abs(right - center) and abs(right - center) < abs(right - left):
            a,b = center,right
        else:
            a,b = left,right
        if TEST:
            print(f"({a},{b})",end="\r")
        
        cropped_image_cp = frame_gpu[(a+b)//2:,:]
        tic6 = time.time()
        ajuste = (a+b)//2
        
        ##### DETECTOR 1 ######
        hsv_cp, cuadros_rgb = detector_rgb_gpu(cropped_image_cp)
        tic7 = time.time()
        ##### DETECTOR 2 ######
        cmeans_image, cmeans_fila_interes, cuadros_cmeans = fcm(frame_gpu,4,punto_horizonte=ajuste-20)
        tic8 = time.time()
        ###### DETECTOR LIDAR ######
        lidar:tuple[float,float] = (3.3,20.2) # Acá se define la función
        
        for cuadro in cuadros_rgb:
            cuadro["y_centroid"] = cuadro["y_centroid"] + ajuste - (a+b)//2
        j = 0
        tic9 = time.time()
        for k in range(len(cuadros_cmeans)):
            index = j + k
            cuadros_cmeans[index]["y_centroid"] = cuadros_cmeans[index]["y_centroid"] + cmeans_fila_interes - (a+b)//2
            if cuadros_cmeans[index]["y_centroid"] < 0:
                del cuadros_cmeans[index]
                j -= 1
        tic10 = time.time()
        
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
        tic11 = time.time()
        ############## IMPLEMENTACIÓN DE MEMORIA ################
        """
            Se utiliza fuzzy_union, y generalmente éste tiene pocos valores, 1 o 2 por frame, por lo que un for no
            va a ser prácticamente una carga para la cpu

        """
        # Agregado a la memoria
        for fuzzy_frame in fuzzy_frames:
            if fuzzy_frame is None: continue
            in_memory = False
            for memory in video_memory:
                if fuzzy_frame["weight"] > 100:
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
                continue
            max_weight = max(max_weight,memory.get_weight())     # Peso (cantidad de pixeles) del obstáculo
            max_score = max(max_score,memory.score())           # Cantidad de frames en los que se detectó el obstáculo
            if final_score < max_weight * (max_score/2):
                final_score = max_weight * (max_score/2)
                x,y = memory.get_point() 
                P1,P2 = memory.get_rectangle()        # Coordenadas del centroide del obstáculo
                y += ajuste
                P1[1] += ajuste
                P2[1] += ajuste
        
        #########################################################
        tic12 = time.time()
        ######## UTILIZACIÓN DE RESULTADOS ###########
        """
           Se puede obtener los puntos x e y de cada obstáculo detectado de la siguiente forma:
           suponiendo que se recorre video_memory

           Para este caso solo se tiene en cuenta el obstáculo con el score más alto, es decir el que tenga el final_score más alto
           x,y: son el centroide del obstáculo que debe ser tomado mayormente en cuenta
        """
        # Resultado te dice si el objeto está a la izquierda, derecha o centro
        resultado = "Libre"
        if max_score >= 5 and max_weight > 200:
            resultado = "Izquierda" if x < max_x//2 else "Derecha" if x > max_x//2 else "Centro"
        ##############################################
        if LIDAR:
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
        
        
        if TEST:
            # Muestra el frame
            if resultado != "Libre":
                cv2.rectangle(image_np,P1,P2,(0,0,255),2)
                # for cuadro in cuadros_rgb:
                #     cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]+ajuste), (cuadro["x_end"],cuadro["y_end"]+ajuste), (0, 0, 255), 2)
                # for cuadro in cuadros_cmeans:
                #     cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]+cmeans_fila_interes), (cuadro["x_end"],cuadro["y_end"]+cmeans_fila_interes), (0, 255, 0), 2)
            cv2.line(image_np,(0,a),(max_x-1,b),(255,0,0),2)
            cv2.imshow('Frame', image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        tic13 = time.time()
        if TEST:
            print(f"""get video stream: {tic2-tic1}s       \n
            read image: {tic3-tic2}s        \n
            separate pixel: {tic4-tic3}s       \n
            find horizon: {tic5-tic4}s         \n
            crop: {tic6-tic5}s       \n
            detector rgb: {tic7-tic6}s         \n
            detector cmeans: {tic8-tic7}s        \n
            cuadros rgb centroides: {tic9-tic8}s         \n
            cuadros cmeans centroides: {tic10-tic9}s          \n
            union: {tic11-tic10}s           \n
            memoria: {tic12-tic11}s           \n
            mostrar: {tic13-tic12}s            \n""",end="\r")
        print(resultado+"        ",end="\r")
        
        if resultado == "libre":
            cambiar_estado_obstaculos([0,0,0])
        if resultado == "centro":
            cambiar_estado_obstaculos([0,1,0])
        if resultado == "izquierda":
            cambiar_estado_obstaculos([0,1,1])
        if resultado == "derecha":
            cambiar_estado_obstaculos([1,1,0])
    if TEST:
        cv2.destroyAllWindows()

    return resultado

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

if __name__ == "__main__":


    # HAY QUE CAMBIAR EL MAIN POR main_video() PARA LAS PRUEBAS FINALES
    resultado = main_video()
    


