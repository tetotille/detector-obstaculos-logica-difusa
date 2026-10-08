"""
Pipeline de Detección Difusa Acelerado por GPU (PyTorch CUDA).

Ejecuta el pipeline completo utilizando la GPU de la NVIDIA Jetson Orin Nano:
Horizonte → RGB adaptativo → FCM en GPU (PyTorch CUDA) → Extracción de regiones → Fusión difusa corregida.

Características:
- Algoritmo Fuzzy C-Means (FCM) totalmente vectorizado en tensores PyTorch sobre GPU CUDA.
- Extracción de regiones y compensación de coordenadas respecto a la imagen completa (256x192).
- Fusión difusa con centroides reales y funciones acotadas [0, 1].
- Sin memoria temporal (detección independiente por cuadro).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.utils import read_image, mask_to_bounding_boxes
from src.detector_horizonte.pixel_detector import separate_pixels, find_largest_fuzzy_jump
from src.detector_hsv.rgb_detection import process_image_cpu as detector_rgb_cpu
from src.fuzzy_union.fuzzy_union import fuzzy_union, intersect_fuzzy_detections, confirm_fuzzy_consensus


def normalize_power_columns_torch(matrix: torch.Tensor, power: float) -> torch.Tensor:
    powered = torch.pow(matrix, power)
    col_sums = torch.sum(powered, dim=0, keepdim=True)
    col_sums = torch.clamp(col_sums, min=1e-10)
    return powered / col_sums


def normalize_columns_torch(u: torch.Tensor) -> torch.Tensor:
    col_sums = torch.sum(u, dim=0, keepdim=True)
    col_sums = torch.clamp(col_sums, min=1e-10)
    return u / col_sums


def cmeans_torch(
    data: torch.Tensor,
    c: int = 4,
    m: float = 2.0,
    error: float = 0.05,
    maxiter: int = 10,
    seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuzzy C-Means clustering acelerado en PyTorch CUDA.
    
    data: Tensor con forma (N, S) donde S es num_features (3 canales RGB) y N es num_pixeles.
    """
    device = data.device
    N, S = data.shape

    if seed is not None:
        torch.manual_seed(seed)
    
    # Inicialización aleatoria normalizada
    u0 = torch.rand(c, N, dtype=torch.float32, device=device)
    u = normalize_columns_torch(u0)
    u = torch.clamp(u, min=1e-7)

    for _ in range(maxiter):
        u_old = u.clone()
        um = torch.pow(u_old, m)
        um_sum = um.sum(dim=1, keepdim=True)
        um_sum = torch.clamp(um_sum, min=1e-10)
        
        # Centros de clusters: (c, S)
        centers = (um @ data) / um_sum

        # Distancias euclídeas exactas L2: (c, N)
        diff = data.unsqueeze(1) - centers.unsqueeze(0)  # (N, c, S)
        dists = torch.norm(diff, dim=2).T                # (c, N)
        dists = torch.clamp(dists, min=1e-6)

        # Actualización de matriz de pertenencia: d ** (-2 / (m - 1))
        u = normalize_power_columns_torch(dists, -2.0 / (m - 1.0))

        if torch.norm(u - u_old) < error:
            break

    return centers, u


def is_valid_obstacle_geometry(width: int, height: int, max_width: int) -> bool:
    if width >= max_width * 0.85:
        return False
    if height >= 12:
        return (width / height) <= 6.5
    elif height >= 8:
        return (width / height) <= 2.8
    return False


def fcm_gpu(
    image_np: np.ndarray,
    num_clusters: int = 4,
    punto_horizonte: int = 0,
    min_contrast: float = 0.245,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, int, List[Dict[str, Any]]]:
    """
    Segmentación FCM en GPU en dos etapas:
    1. Segmentación semántica de agua y obstáculos a nivel de píxel (sin encuadrar primero).
    2. Algoritmo de encuadre aplicado posteriormente sobre la máscara de obstáculos limpia.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    h, w, c = image_np.shape
    y_nav_start = max(0, min(punto_horizonte + 5, h - 20))
    y_nav_end = min(174, h)
    
    water_np = image_np[y_nav_start:y_nav_end, :]
    H_w, W_w, _ = water_np.shape
    if H_w < 5 or W_w < 5:
        empty = np.zeros((H_w, W_w), dtype=np.uint8)
        return empty, empty, punto_horizonte, []

    # 1. Normalización de fondo / supresión de degradado vertical
    water_filtered = cv2.bilateralFilter(water_np, 5, 50, 50)
    row_med = np.median(water_filtered, axis=1, keepdims=True)
    row_smooth = cv2.GaussianBlur(row_med.astype(np.float32), (1, 15), 0)
    res = water_filtered.astype(np.float32) - row_smooth + 128.0
    res_norm = np.clip(res, 0, 255) / 255.0

    # 2. Transferir a GPU y ejecutar FCM
    water_t = torch.from_numpy(res_norm).to(dev, dtype=torch.float32)
    data = water_t.view(-1, c)  # (H_w * W_w, 3)

    centers, u = cmeans_torch(data, c=num_clusters, maxiter=20)
    labels = torch.argmax(u, dim=0).view(H_w, W_w)

    # 3. Encontrar cluster dominante de agua (el más cercano al residuo neutro 0.5)
    neutral_t = torch.tensor([0.5, 0.5, 0.5], device=dev)
    dists_neutral = torch.norm(centers - neutral_t, dim=1)
    water_cluster_id = torch.argmin(dists_neutral).item()

    # 4. Cluster minoritario candidato a obstáculo
    counts = [int((labels == c_id).sum().item()) for c_id in range(num_clusters)]
    min_c_id = int(np.argmin(counts))
    contrast_min = torch.norm(centers[min_c_id] - centers[water_cluster_id]).item()

    obstacle_mask_t = torch.zeros((H_w, W_w), dtype=torch.uint8, device=dev)
    water_mask_t = torch.zeros((H_w, W_w), dtype=torch.uint8, device=dev)

    if contrast_min >= min_contrast and counts[min_c_id] < (H_w * W_w * 0.25):
        obstacle_mask_t[labels == min_c_id] = 255
        water_mask_t[labels != min_c_id] = 255
    else:
        water_mask_t[:, :] = 255

    obstacle_mask_np = obstacle_mask_t.cpu().numpy()
    water_mask_np = water_mask_t.cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obstacle_mask_np = cv2.morphologyEx(obstacle_mask_np, cv2.MORPH_OPEN, kernel)

    # 5. Algoritmo de encuadre posterior sobre la máscara limpia
    raw_boxes = mask_to_bounding_boxes(obstacle_mask_np, min_area=25)
    formatted_boxes = []
    for b in raw_boxes:
        w_b = b["x_end"] - b["x_init"]
        h_b = b["y_end"] - b["y_init"]
        if is_valid_obstacle_geometry(w_b, h_b, W_w):
            c_box = dict(b)
            c_box["y_init"] = int(b["y_init"] + y_nav_start)
            c_box["y_end"] = int(b["y_end"] + y_nav_start)
            c_box["y_centroid"] = int(b["y_centroid"] + y_nav_start)
            c_box["_compensated"] = True
            formatted_boxes.append(c_box)

    full_obs_mask = np.zeros((h, w), dtype=np.uint8)
    full_obs_mask[y_nav_start:y_nav_end, :] = obstacle_mask_np
    full_water_mask = np.zeros((h, w), dtype=np.uint8)
    full_water_mask[y_nav_start:y_nav_end, :] = water_mask_np

    return full_water_mask, full_obs_mask, punto_horizonte, formatted_boxes


def detect_obstacles_gpu(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (256, 192),
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Pipeline completo acelerado por GPU:
    Horizonte → Segmentación FCM (GPU) → Extracción de Regiones → Fusión difusa e Intersección.
    """
    w, h = target_size
    image_np, _ = read_image(frame, w, h)

    # 1. Detección de Horizonte
    left, center, right = separate_pixels(image_np)
    h_left = find_largest_fuzzy_jump(left)
    h_center = find_largest_fuzzy_jump(center)
    h_right = find_largest_fuzzy_jump(right)

    if abs(h_center - h_left) < abs(h_right - h_center) and abs(h_center - h_left) < abs(h_right - h_left):
        hz_a, hz_b = h_left, h_center
    elif abs(h_center - h_left) > abs(h_right - h_center) and abs(h_right - h_center) < abs(h_right - h_left):
        hz_a, hz_b = h_center, h_right
    else:
        hz_a, hz_b = h_left, h_right

    ajuste = int((hz_a + hz_b) // 2)
    ajuste = max(0, min(ajuste, h - 25))

    y_nav_start = max(0, min(ajuste + 5, h - 20))
    y_nav_end = min(174, h)
    water_roi = image_np[y_nav_start:y_nav_end, :]

    # 2. FCM en GPU (PyTorch CUDA)
    m_agua, m_obs, _, cuadros_fcm = fcm_gpu(
        image_np,
        num_clusters=4,
        punto_horizonte=ajuste,
        min_contrast=0.245,
        device=device
    )

    # 3. Detector Cromático
    hsv_roi = cv2.cvtColor(water_roi, cv2.COLOR_BGR2HSV)
    med_v = float(np.median(hsv_roi[:, :, 2]))
    color_anomaly = (hsv_roi[:, :, 1] > 36) | (np.abs(hsv_roi[:, :, 2].astype(float) - med_v) > 42)
    kernel_m = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_col_roi = cv2.morphologyEx(color_anomaly.astype(np.uint8), cv2.MORPH_OPEN, kernel_m)
    m_col_roi = cv2.morphologyEx(m_col_roi, cv2.MORPH_CLOSE, kernel_m)

    raw_col = mask_to_bounding_boxes(m_col_roi, min_area=20)
    cuadros_color = []
    for b in raw_col:
        w_b = b["x_end"] - b["x_init"]
        h_b = b["y_end"] - b["y_init"]
        if is_valid_obstacle_geometry(w_b, h_b, water_roi.shape[1]):
            c_box = dict(b)
            c_box["y_init"] = int(b["y_init"] + y_nav_start)
            c_box["y_end"] = int(b["y_end"] + y_nav_start)
            c_box["y_centroid"] = int(b["y_centroid"] + y_nav_start)
            c_box["_compensated"] = True
            cuadros_color.append(c_box)

    # 4. Fusión Difusa (Puntuación y selección de candidatos por rama)
    fused_boxes = fuzzy_union([cuadros_color, cuadros_fcm])

    # 5. Confirmación por CONSENSO DE FUSIÓN DIFUSA (ramas distintas + puntuación > 0 + solapamiento)
    confirmed_candidates = confirm_fuzzy_consensus(fused_boxes, tol=0)

    confirmed_boxes = []
    for box in confirmed_candidates:
        bx1, by1 = int(box["x_init"]), int(box["y_init"])
        bx2, by2 = int(box["x_end"]), int(box["y_end"])
        crop = image_np[by1:by2, bx1:bx2]
        if crop.size > 0:
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s_p90 = float(np.percentile(crop_hsv[:, :, 1], 90))
            if s_p90 >= 55.0 or (box.get("height", 0) >= 12 and float(np.abs(crop_hsv[:, :, 2].mean() - med_v)) > 35.0):
                confirmed_boxes.append(box)

    return {
        "confirmed_boxes": confirmed_boxes,
        "has_obstacle": len(confirmed_boxes) > 0,
        "horizon": {"a": int(hz_a), "b": int(hz_b), "ajuste": int(ajuste)},
        "boxes_rgb": cuadros_color,
        "boxes_fcm": cuadros_fcm,
        "mask_agua": m_agua,
        "mask_obstaculos": m_obs
    }

