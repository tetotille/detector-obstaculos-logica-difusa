"""
Pipeline de Detección de Obstáculos Basado en Lógica Difusa.

Pipeline secuencial:
Horizonte → RGB adaptativo → FCM con 4 clusters → Extracción de regiones → Fusión difusa.

Características:
- Sin incorporación de memoria temporal (detección cuadro a cuadro independiente).
- Coordenadas de RGB y FCM compensadas hacia el marco completo de la imagen (256x192).
- Fusión difusa corregida con centroides reales, funciones acotadas [0, 1] y distancias finitas.
- Ejecución pura sin visualización ni despliegue de ventanas gráficas (headless).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

# Configuración del entorno y sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.utils import read_image
from src.detector_horizonte.pixel_detector import separate_pixels, find_largest_fuzzy_jump
from src.detector_hsv.rgb_detection import process_image_cpu as detector_rgb_cpu
from src.cmeans.c_means_main import fcm, segment_fcm_pixel_level, extract_boxes_from_mask
from src.fuzzy_union.fuzzy_union import fuzzy_union, intersect_fuzzy_detections, confirm_fuzzy_consensus



def detect_obstacles(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (256, 192)
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline de detección difusa sobre un frame individual sin visualización.

    Parámetros:
    frame: np.ndarray
        Imagen BGR de entrada (cualquier resolución).
    target_size: tuple (width, height)
        Resolución de procesamiento estándar del algoritmo (por defecto 256x192).

    Retorna:
    dict con los resultados de la detección:
        - "confirmed_boxes": lista de cuadros delimitadores de obstáculos confirmados por la fusión.
        - "has_obstacle": bool (True si hay al menos un obstáculo con puntuación ponderada > 0).
        - "horizon": dict con posición de línea de horizonte (a, b, ajuste).
        - "boxes_rgb": cuadros detectados por el módulo RGB (compensados a imagen completa).
        - "boxes_fcm": cuadros detectados por el módulo FCM (compensados a imagen completa).
    """
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido o vacío proporcionado a detect_obstacles.")

    w, h = target_size
    # Redimensionar al tamaño estándar de procesamiento
    image_np, image_cp = read_image(frame, w, h)

    # 1. Detección de la línea de horizonte
    left, center, right = separate_pixels(image_np)
    h_left = find_largest_fuzzy_jump(left)
    h_center = find_largest_fuzzy_jump(center)
    h_right = find_largest_fuzzy_jump(right)

    # Selección de los puntos más consistentes para la recta de horizonte
    if abs(h_center - h_left) < abs(h_right - h_center) and abs(h_center - h_left) < abs(h_right - h_left):
        hz_a, hz_b = h_left, h_center
    elif abs(h_center - h_left) > abs(h_right - h_center) and abs(h_right - h_center) < abs(h_right - h_left):
        hz_a, hz_b = h_center, h_right
    else:
        hz_a, hz_b = h_left, h_right

    ajuste = int((hz_a + hz_b) // 2)
    # Asegurar que el recorte esté dentro de los límites válidos de la imagen
    ajuste = max(0, min(ajuste, h - 25))

    # Región navegable útil: excluye la franja de orilla/costa bajo el horizonte
    # y la proa del barco propio en la base de la cámara (y >= 174)
    y_nav_start = max(0, min(ajuste + 5, h - 20))
    y_nav_end = min(174, h)
    water_roi = image_np[y_nav_start:y_nav_end, :]
    H_w, W_w, _ = water_roi.shape

    if H_w < 10 or W_w < 10:
        return {
            "confirmed_boxes": [],
            "has_obstacle": False,
            "horizon": {"a": int(hz_a), "b": int(hz_b), "ajuste": int(ajuste)},
            "boxes_rgb": [],
            "boxes_fcm": [],
            "mask_agua": np.zeros((h, w), dtype=np.uint8),
            "mask_obstaculos": np.zeros((h, w), dtype=np.uint8)
        }

    # 2. Detector FCM en dos etapas:
    # 2.1. Segmentación semántica a nivel de píxel (agua y obstáculos) SIN encuadrar primero
    mask_agua_roi, mask_obs_roi, fcm_info = segment_fcm_pixel_level(
        water_roi,
        num_clusters=4,
        m=2.0,
        min_contrast=0.245
    )

    # Máscaras semánticas completas referenciadas al cuadro global (256x192)
    mask_agua = np.zeros((h, w), dtype=np.uint8)
    mask_agua[y_nav_start:y_nav_end, :] = mask_agua_roi
    mask_obstaculos = np.zeros((h, w), dtype=np.uint8)
    mask_obstaculos[y_nav_start:y_nav_end, :] = mask_obs_roi

    # 2.2. Algoritmo de encuadre aplicado posteriormente sobre la máscara de obstáculos limpia
    # Filtra estelas planas de oleaje (h < 8 px) y compensa coordenadas al marco global
    cuadros_fcm = extract_boxes_from_mask(
        mask_obs_roi,
        y_offset=y_nav_start,
        min_area=25
    )

    # 3. Detector Complementario Cromático / Fusión Difusa (Color, Saturación y Contraste)
    hsv_roi = cv2.cvtColor(water_roi, cv2.COLOR_BGR2HSV)
    med_v = float(np.median(hsv_roi[:, :, 2]))
    # En lagos, el agua es neutra/grisácea (S < 30). Un obstáculo real posee saturación cromática
    # o contraste fuerte de luminosidad frente al fondo lacustre
    color_anomaly = (hsv_roi[:, :, 1] > 36) | (np.abs(hsv_roi[:, :, 2].astype(float) - med_v) > 42)
    kernel_m = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_col_roi = cv2.morphologyEx(color_anomaly.astype(np.uint8), cv2.MORPH_OPEN, kernel_m)
    m_col_roi = cv2.morphologyEx(m_col_roi, cv2.MORPH_CLOSE, kernel_m)

    cuadros_color = extract_boxes_from_mask(
        m_col_roi,
        y_offset=y_nav_start,
        min_area=20
    )

    # 4. Fusión difusa (Fuzzy Union)
    # Evalúa reglas difusas y asigna puntuaciones según distancias espaciales y masas
    fused_boxes = fuzzy_union([cuadros_color, cuadros_fcm])

    # 5. Confirmación por CONSENSO DE FUSIÓN DIFUSA
    # Exige acuerdo entre ramas diferentes (color y FCM), con puntuación > 0 y solapamiento 2D
    confirmed_candidates = confirm_fuzzy_consensus(fused_boxes, tol=0)

    confirmed_boxes = []
    for box in confirmed_candidates:
        bx1, by1 = int(box["x_init"]), int(box["y_init"])
        bx2, by2 = int(box["x_end"]), int(box["y_end"])
        crop = image_np[by1:by2, bx1:bx2]
        if crop.size > 0:
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s_p90 = float(np.percentile(crop_hsv[:, :, 1], 90))
            # Validación física: cuerpo con cromatismo o fuerte contraste con compacidad
            if s_p90 >= 55.0 or (box.get("height", 0) >= 12 and float(np.abs(crop_hsv[:, :, 2].mean() - med_v)) > 35.0):
                confirmed_boxes.append(box)

    return {
        "confirmed_boxes": confirmed_boxes,
        "has_obstacle": len(confirmed_boxes) > 0,
        "horizon": {
            "a": int(hz_a),
            "b": int(hz_b),
            "ajuste": int(ajuste)
        },
        "boxes_rgb": cuadros_color,
        "boxes_fcm": cuadros_fcm,
        "mask_agua": mask_agua,
        "mask_obstaculos": mask_obstaculos
    }



def process_video_headless(
    video_path: str,
    max_frames: Optional[int] = None,
    step: int = 1
) -> Dict[str, Any]:
    """
    Procesa un archivo de video completo secuencialmente sin visualización.
    
    Retorna métricas de tiempo y detecciones por frame.
    """
    import time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video en {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    evaluated_count = 0
    detections_log = {}
    frame_times = []

    print(f"[+] Iniciando pipeline difuso headless sobre: {video_path}")
    print(f"    Total frames en video: {total_video_frames} | Step: {step}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t0 = time.perf_counter()
            result = detect_obstacles(frame)
            t_elapsed = time.perf_counter() - t0

            frame_times.append(t_elapsed)
            detections_log[frame_idx] = {
                "has_obstacle": result["has_obstacle"],
                "num_boxes": len(result["confirmed_boxes"]),
                "boxes": [
                    {
                        "x_init": b["x_init"],
                        "y_init": b["y_init"],
                        "x_end": b["x_end"],
                        "y_end": b["y_end"],
                        "score": b.get("fuzzy_union", 0.0),
                        "weight": b.get("weight", 0)
                    }
                    for b in result["confirmed_boxes"]
                ],
                "horizon_ajuste": result["horizon"]["ajuste"],
                "time_sec": t_elapsed
            }
            evaluated_count += 1

            if evaluated_count % 50 == 0:
                print(f"    Evaluados {evaluated_count} frames... Último: {t_elapsed*1000:.1f} ms")

            if max_frames is not None and evaluated_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    # Cálculo de tiempos excluyendo calentamiento si hay suficientes frames
    warmup_n = min(5, len(frame_times) // 5)
    timed_frames = frame_times[warmup_n:] if len(frame_times) > warmup_n else frame_times
    mean_time = float(np.mean(timed_frames)) if timed_frames else 0.0
    fps = (1.0 / mean_time) if mean_time > 0 else 0.0

    print(f"[✓] Procesamiento finalizado:")
    print(f"    Frames evaluados: {evaluated_count}")
    print(f"    Tiempo medio: {mean_time*1000:.2f} ms | Throughput: {fps:.2f} FPS")

    return {
        "video_path": str(video_path),
        "total_evaluated": evaluated_count,
        "mean_time_sec": mean_time,
        "fps": fps,
        "detections": detections_log
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline difuso sin visualización")
    parser.add_argument("--video", type=str, default="assets/videos/tesis.mp4", help="Ruta al video")
    parser.add_argument("--max_frames", type=int, default=100, help="Máximo de frames a evaluar")
    parser.add_argument("--step", type=int, default=10, help="Paso de frames")
    args = parser.parse_args()

    v_path = Path(args.video)
    if not v_path.is_absolute():
        v_path = project_root / v_path

    if v_path.exists():
        process_video_headless(str(v_path), max_frames=args.max_frames, step=args.step)
    else:
        print(f"Video no encontrado en: {v_path}")
