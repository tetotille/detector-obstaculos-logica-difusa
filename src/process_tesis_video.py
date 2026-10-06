import sys
import os

# Añadir el directorio raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import cv2
from os.path import dirname, abspath, join
from src.cmeans import fcm
from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv import detector_rgb
from src.utils import read_image, FrameMemory
from src.fuzzy_union.fuzzy_union import fuzzy_union
import time

def process_video(video_path):
    print(f"Iniciando procesamiento de video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return

    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Definir el codec y crear el objeto VideoWriter para guardar el resultado
    output_path = "main_output/tesis_procesado.mp4"
    os.makedirs("main_output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Usaremos el tamaño al que redimensionamos para el procesamiento o el original?
    # Para consistencia visual, usaremos el tamaño de procesamiento (256, 192) o redimensionaremos de vuelta.
    # Vamos a guardar en 256x192 para que sea rápido.
    out = cv2.VideoWriter(output_path, fourcc, fps, (256, 192))

    frame_count = 0
    display_available = "DISPLAY" in os.environ
    if display_available:
        print("Display detectado. Mostrando procesamiento en tiempo real...")
    else:
        print("No se detectó entorno gráfico (DISPLAY). El procesamiento continuará sin visualización.")

    # Inicialización de memoria temporal (5 frames de contexto)
    video_memory: list[FrameMemory] = []
    memory_limit = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        init = time.time()

        # ... (resto del procesamiento igual) ...
        # [Insertando lógica de visualización después del dibujo]
        
        # Redimensionar y preparar imagen (256x192 es lo que usa main_cpu)
        image_np, image_cp = read_image(frame, 256, 192)
        
        # 1. Detección de Horizonte
        left, center, right = separate_pixels(image_np)
        h_left = find_largest_fuzzy_jump(left)
        h_center = find_largest_fuzzy_jump(center)
        h_right = find_largest_fuzzy_jump(right)

        # Selección de línea de horizonte (lógica de main_cpu)
        if abs(h_center - h_left) < abs(h_right - h_center) and abs(h_center - h_left) < abs(h_right - h_left):
            a, b = h_left, h_center
        elif abs(h_center - h_left) > abs(h_right - h_center) and abs(h_right - h_center) < abs(h_right - h_left):
            a, b = h_center, h_right
        else:
            a, b = h_left, h_right
        
        ajuste = (a + b) // 2
        
        # 2. Recorte debajo del horizonte
        cropped_image_np = image_np[ajuste:, :]
        cropped_image_cp = image_cp[ajuste:, :]

        # 3. Detector RGB/HSV
        hsv_np, cuadros_rgb = detector_rgb(cropped_image_cp)
        
        # 4. Detector C-Means
        cmeans_image, cmeans_fila_interes, cuadros_cmeans = fcm(image_np, 4)

        # 5. Ajuste de coordenadas de los cuadros
        for cuadro in cuadros_rgb:
            cuadro["y_init"] += ajuste
            cuadro["y_end"] += ajuste
            cuadro["y_centroid"] += ajuste

        j = 0
        for k in range(len(cuadros_cmeans)):
            idx = j + k
            cuadros_cmeans[idx]["y_init"] += cmeans_fila_interes
            cuadros_cmeans[idx]["y_end"] += cmeans_fila_interes
            cuadros_cmeans[idx]["y_centroid"] += cmeans_fila_interes
            if cuadros_cmeans[idx]["y_centroid"] < 0:
                del cuadros_cmeans[idx]
                j -= 1

        # 6. Unión Difusa
        fuzzy_frames = fuzzy_union([cuadros_rgb, cuadros_cmeans])

        # 7. Memoria temporal de 5 frames (FrameMemory)
        for fuzzy_frame in fuzzy_frames:
            if fuzzy_frame is None: continue
            in_memory = False
            for memory in video_memory:
                if fuzzy_frame in memory:
                    memory.add(fuzzy_frame)
                    memory.modified = True
                    in_memory = True
            if not in_memory:
                memory = FrameMemory(memory_limit, fuzzy_frame)
                video_memory.append(memory)

        confirmed_boxes = []
        for memory in video_memory[:]:
            if not memory.modified:
                memory.add(None)
            memory.modified = False
            if memory.empty():
                video_memory.remove(memory)
                continue
            # Obstáculo confirmado si persiste al menos 2 de los últimos 5 frames
            if memory.score() >= 2:
                P1, P2 = memory.get_rectangle()
                confirmed_boxes.append({
                    "x_init": P1[0], "y_init": P1[1],
                    "x_end": P2[0], "y_end": P2[1],
                    "score": memory.score(),
                    "weight": memory.get_weight()
                })

        # 8. Dibujar resultados en el frame (image_np ya está en 256x192)
        cv2.line(image_np, (0, a), (image_np.shape[1] // 2, a), (255, 0, 0), 2)
        cv2.line(image_np, (image_np.shape[1] // 2, b), (image_np.shape[1], b), (255, 0, 0), 2)

        # Dibujar obstáculos confirmados por persistencia temporal (en rojo)
        for cuadro in confirmed_boxes:
            cv2.rectangle(image_np, (cuadro["x_init"], cuadro["y_init"]), 
                         (cuadro["x_end"], cuadro["y_end"]), (0, 0, 255), 2)
            cv2.putText(image_np, f"Obs:{cuadro['score']}/5", (cuadro["x_init"], max(15, cuadro["y_init"] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Mostrar en tiempo real si hay display
        if display_available:
            cv2.imshow("Procesamiento en Tiempo Real", image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Escribir frame al video de salida
        out.write(image_np)

        if frame_count % 10 == 0:
            print(f"Frame {frame_count} procesado. Tiempo: {time.time() - init:.3f}s")

    cap.release()
    out.release()
    if display_available:
        cv2.destroyAllWindows()
    print(f"Procesamiento finalizado. Video guardado en: {output_path}")

if __name__ == "__main__":
    video_file = "/home/tetotille/Proyectos/detector-obstaculos-logica-difusa/assets/videos/tesis.mp4"
    process_video(video_file)
