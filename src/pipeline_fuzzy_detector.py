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
from src.cmeans.c_means_main import fcm
from src.fuzzy_union.fuzzy_union import fuzzy_union


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
        a, b = h_left, h_center
    elif abs(h_center - h_left) > abs(h_right - h_center) and abs(h_right - h_center) < abs(h_right - h_left):
        a, b = h_center, h_right
    else:
        a, b = h_left, h_right

    ajuste = int((a + b) // 2)
    # Asegurar que el recorte esté dentro de los límites válidos de la imagen
    ajuste = max(0, min(ajuste, h - 10))

    # 2. Detector RGB Adaptativo sobre el área navegable (debajo del horizonte)
    cropped_image_np = image_np[ajuste:, :]
    _, cuadros_rgb_raw = detector_rgb_cpu(cropped_image_np)

    # Compensar coordenadas de RGB hacia la imagen completa
    cuadros_rgb = []
    for c in cuadros_rgb_raw:
        c_comp = dict(c)
        c_comp["y_init"] = int(c["y_init"] + ajuste)
        c_comp["y_end"] = int(c["y_end"] + ajuste)
        c_comp["y_centroid"] = int(c["y_centroid"] + ajuste)
        c_comp["_compensated"] = True
        cuadros_rgb.append(c_comp)

    # 3. Detector FCM con 4 clusters sobre la imagen completa recortando bajo horizonte
    _, cmeans_fila_interes, cuadros_cmeans_raw = fcm(
        image_np,
        num_clusters=4,
        punto_horizonte=ajuste
    )

    # Compensar coordenadas de FCM hacia la imagen completa
    cuadros_cmeans = []
    for c in cuadros_cmeans_raw:
        c_comp = dict(c)
        c_comp["y_init"] = int(c["y_init"] + ajuste)
        c_comp["y_end"] = int(c["y_end"] + ajuste)
        c_comp["y_centroid"] = int(c["y_centroid"] + ajuste)
        c_comp["_compensated"] = True
        cuadros_cmeans.append(c_comp)

    # 4. Fusión difusa (Fuzzy Union)
    # Recibe ambos conjuntos de candidatos ya compensados
    fused_boxes_raw = fuzzy_union([cuadros_rgb, cuadros_cmeans])

    # 5. Filtrado de obstáculos confirmados (sin memoria temporal)
    # Un obstáculo es confirmado si la fusión difusa asignó una puntuación ponderada > 0
    confirmed_boxes = []
    for box in fused_boxes_raw:
        if box is not None and box.get("fuzzy_union", 0.0) > 0.0:
            confirmed_boxes.append(box)

    return {
        "confirmed_boxes": confirmed_boxes,
        "has_obstacle": len(confirmed_boxes) > 0,
        "horizon": {
            "a": int(a),
            "b": int(b),
            "ajuste": int(ajuste)
        },
        "boxes_rgb": cuadros_rgb,
        "boxes_fcm": cuadros_cmeans
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
