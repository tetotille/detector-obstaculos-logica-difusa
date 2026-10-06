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
from src.fuzzy_union.fuzzy_union import fuzzy_union


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


def fcm_gpu(
    image_np: np.ndarray,
    num_clusters: int = 4,
    punto_horizonte: int = 0,
    min_contrast: float = 0.20,
    device: str = "cuda"
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    """
    Segmentación FCM en GPU en dos etapas:
    1. Segmentación semántica de agua y obstáculos a nivel de píxel (sin encuadrar primero).
    2. Algoritmo de encuadre aplicado posteriormente sobre la máscara de obstáculos limpia.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    h, w, c = image_np.shape
    punto_horizonte = max(0, min(punto_horizonte, h - 5))
    
    water_np = image_np[punto_horizonte:, :]
    H_w, W_w, _ = water_np.shape
    if H_w < 5 or W_w < 5:
        empty = np.zeros((H_w, W_w), dtype=np.uint8)
        return empty, punto_horizonte, []

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

    # 4. Máscara semántica de obstáculos a nivel de píxel
    obstacle_mask_t = torch.zeros((H_w, W_w), dtype=torch.uint8, device=dev)
    for c_id in range(num_clusters):
        if c_id == water_cluster_id:
            continue
        contrast = torch.norm(centers[c_id] - centers[water_cluster_id]).item()
        if contrast >= min_contrast:
            obstacle_mask_t[labels == c_id] = 255

    obstacle_mask_np = obstacle_mask_t.cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obstacle_mask_np = cv2.morphologyEx(obstacle_mask_np, cv2.MORPH_OPEN, kernel)

    # 5. Algoritmo de encuadre posterior sobre la máscara limpia
    boxes = mask_to_bounding_boxes(obstacle_mask_np, min_area=35)
    formatted_boxes = []
    for b in boxes:
        formatted_boxes.append({
            "puntos": None,
            "x_init": b["x_init"],
            "x_end": b["x_end"],
            "y_init": b["y_init"],
            "y_end": b["y_end"],
            "x_centroid": b["x_centroid"],
            "y_centroid": b["y_centroid"],
            "weight": b["weight"],
            "x": b["x_centroid"] * b["weight"],
            "y": b["y_centroid"] * b["weight"]
        })

    return obstacle_mask_np, punto_horizonte, formatted_boxes



def detect_obstacles_gpu(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (256, 192),
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Pipeline completo acelerado por GPU:
    Horizonte → RGB adaptativo → FCM (GPU CUDA) → Fusión difusa corregida.
    """
    w, h = target_size
    image_np, _ = read_image(frame, w, h)

    # 1. Detección de Horizonte
    left, center, right = separate_pixels(image_np)
    h_left = find_largest_fuzzy_jump(left)
    h_center = find_largest_fuzzy_jump(center)
    h_right = find_largest_fuzzy_jump(right)

    if abs(h_center - h_left) < abs(h_right - h_center) and abs(h_center - h_left) < abs(h_right - h_left):
        a, b = h_left, h_center
    elif abs(h_center - h_left) > abs(h_right - h_center) and abs(h_right - h_center) < abs(h_right - h_left):
        a, b = h_center, h_right
    else:
        a, b = h_left, h_right

    ajuste = int((a + b) // 2)
    ajuste = max(0, min(ajuste, h - 10))

    # 2. RGB Adaptativo
    cropped_image_np = image_np[ajuste:, :]
    _, cuadros_rgb_raw = detector_rgb_cpu(cropped_image_np)

    cuadros_rgb = []
    for c in cuadros_rgb_raw:
        c_comp = dict(c)
        c_comp["y_init"] = int(c["y_init"] + ajuste)
        c_comp["y_end"] = int(c["y_end"] + ajuste)
        c_comp["y_centroid"] = int(c["y_centroid"] + ajuste)
        c_comp["_compensated"] = True
        cuadros_rgb.append(c_comp)

    # 3. FCM en GPU (PyTorch CUDA) en dos etapas
    obs_mask_gpu, _, cuadros_cmeans_raw = fcm_gpu(
        image_np,
        num_clusters=4,
        punto_horizonte=ajuste,
        min_contrast=0.20,
        device=device
    )

    cuadros_cmeans = []
    for c in cuadros_cmeans_raw:
        c_comp = dict(c)
        c_comp["y_init"] = int(c["y_init"] + ajuste)
        c_comp["y_end"] = int(c["y_end"] + ajuste)
        c_comp["y_centroid"] = int(c["y_centroid"] + ajuste)
        c_comp["_compensated"] = True
        cuadros_cmeans.append(c_comp)

    # 4. Fusión Difusa corregida
    fused_boxes = fuzzy_union([cuadros_rgb, cuadros_cmeans])

    confirmed_boxes = [b for b in fused_boxes if b is not None and b.get("fuzzy_union", 0.0) > 0.0]

    return {
        "confirmed_boxes": confirmed_boxes,
        "has_obstacle": len(confirmed_boxes) > 0,
        "horizon": {"a": int(a), "b": int(b), "ajuste": int(ajuste)},
        "boxes_rgb": cuadros_rgb,
        "boxes_fcm": cuadros_cmeans,
        "mask_obstaculos": obs_mask_gpu
    }

