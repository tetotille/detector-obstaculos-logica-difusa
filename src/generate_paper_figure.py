import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cmeans import fcm
from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv.rgb_detection import process_image_cpu as detector_ours
from src.utils.utils import read_image

def generate_qualitative_figure(video_path, output_name="fig_qual.png"):
    print(f"Generando figura cualitativa desde: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Saltamos al frame 150 (un frame interesante con obstáculos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: No se pudo leer el frame del video.")
        return

    # 1. Imagen Original (Redimensionada para el pipeline)
    img_np, _ = read_image(frame, 256, 192)
    img_orig_viz = img_np.copy()

    # 2. Procesamiento de Horizonte
    left, center, right = separate_pixels(img_np)
    h_left = find_largest_fuzzy_jump(left)
    h_center = find_largest_fuzzy_jump(center)
    h_right = find_largest_fuzzy_jump(right)
    
    # Horizonte visual
    img_horizon = img_np.copy()
    cv2.line(img_horizon, (0, h_left), (img_np.shape[1]//2, h_center), (255, 0, 0), 2)
    cv2.line(img_horizon, (img_np.shape[1]//2, h_center), (img_np.shape[1], h_right), (255, 0, 0), 2)
    ajuste = (h_left + h_center + h_right) // 3

    # 3. Segmentación (Máscara Intermedia)
    mask_cmeans, _, _ = fcm(img_np, 4, punto_horizonte=ajuste)
    # Convertimos la máscara binaria a BGR para concatenar
    mask_viz_raw = cv2.cvtColor(mask_cmeans, cv2.COLOR_GRAY2BGR)
    
    # IMPORTANTE: mask_cmeans tiene la altura recortada (192 - ajuste)
    # Debemos rellenar la parte superior con negro para que mida 192 de alto
    mask_viz = np.zeros_like(img_np)
    mask_viz[ajuste:, :] = mask_viz_raw

    # 4. Resultado Final (Detección y Bounding Boxes)
    cropped_img = img_np[ajuste:, :]
    _, cuadros_rgb = detector_ours(cropped_img)
    img_final = img_np.copy()
    
    # Dibujar horizonte en azul (255, 0, 0) y cuadros en rojo (0, 0, 255)
    cv2.line(img_final, (0, h_left), (img_np.shape[1]//2, h_center), (255, 0, 0), 2)
    cv2.line(img_final, (img_np.shape[1]//2, h_center), (img_np.shape[1], h_right), (255, 0, 0), 2)

    for c in cuadros_rgb:
        cv2.rectangle(img_final, (c["x_init"], c["y_init"]+ajuste), 
                     (c["x_end"], c["y_end"]+ajuste), (0, 0, 255), 2)

    # --- ENSAMBLAJE DE LA FIGURA ---
    # Queremos una fila de 3 imágenes: Original, Segmentación, Resultado Final
    # Añadimos etiquetas de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_orig_viz, ' (a) Original', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mask_viz, ' (b) FCM Segmentation', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_final, ' (c) Detection Result', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Concatenar horizontalmente
    combined = np.hstack((img_orig_viz, mask_viz, img_final))

    # Guardar
    cv2.imwrite(output_name, combined)
    print(f"Figura guardada exitosamente como: {output_name}")

if __name__ == "__main__":
    video = "assets/videos/tesis.mp4"
    generate_qualitative_figure(video)
