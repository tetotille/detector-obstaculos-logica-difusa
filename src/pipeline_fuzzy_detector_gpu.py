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

        # Distancias euclídeas: (c, N)
        # data: (N, 1, S), centers: (1, c, S)
        diff = data.unsqueeze(1) - centers.unsqueeze(0)  # (N, c, S)
        dists = torch.sum(diff ** 2, dim=2).T            # (c, N)
        dists = torch.clamp(dists, min=1e-7)

        # Actualización de matriz de pertenencia
        u = normalize_power_columns_torch(dists, -2.0 / (m - 1.0))

        if torch.norm(u - u_old) < error:
            break

    return centers, u


def fcm_gpu(
    image_np: np.ndarray,
    num_clusters: int = 4,
    punto_horizonte: int = 0,
    device: str = "cuda"
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    """
    Segmentación FCM en GPU y extracción de regiones bajo el horizonte.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    h, w, c = image_np.shape

    # Normalizar a [0, 1] en GPU
    img_t = torch.from_numpy(image_np).to(dev, dtype=torch.float32) / 255.0
    data = img_t.view(-1, c)  # (h*w, 3)

    _, u = cmeans_torch(data, c=num_clusters, maxiter=10)
    
    # Asignación de clusters (argmax)
    cluster_membership = torch.argmax(u, dim=0).view(h, w)  # (h, w)
    
    # Recorte bajo el horizonte (zona de agua)
    water_membership = cluster_membership[punto_horizonte:, :]
    
    # El cluster minoritario en el área de agua corresponde al obstáculo
    unique_clusters, counts = torch.unique(water_membership, return_counts=True)
    if len(counts) > 1:
        min_idx = torch.argmin(counts)
        obstacle_cluster = unique_clusters[min_idx].item()
    else:
        obstacle_cluster = unique_clusters[0].item()

    # Generar máscara binaria del obstáculo
    obstacle_mask_t = (water_membership == obstacle_cluster).to(torch.uint8) * 255
    obstacle_mask_np = obstacle_mask_t.cpu().numpy()

    # Extracción de cuadros delimitadores sobre la máscara recortada
    boxes = mask_to_bounding_boxes(obstacle_mask_np, min_area=30)
    
    # Formatear cuadros con campos requeridos por fuzzy_union
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

    # 3. FCM en GPU (PyTorch CUDA)
    _, _, cuadros_cmeans_raw = fcm_gpu(
        image_np,
        num_clusters=4,
        punto_horizonte=ajuste,
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
        "boxes_fcm": cuadros_cmeans
    }
