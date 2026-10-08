import json
import cv2
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.detector_horizonte.pixel_detector import separate_pixels, find_largest_fuzzy_jump
from src.utils.utils import mask_to_bounding_boxes
from src.fuzzy_union.fuzzy_union import fuzzy_union, intersect_fuzzy_detections
from src.detector_hsv.rgb_detection import process_image_cpu as detector_rgb_cpu

# Cargar las 148 anotaciones validadas por el usuario
VALIDATION_FILE = project_root / "main_output/wasrt_reference_validation.json"
with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
    user_val = json.load(f)["annotations"]

approved_frames = sorted([int(k) for k, v in user_val.items() if v is True])

def get_horizon(img_256):
    l, c, r = separate_pixels(img_256)
    hl, hc, hr = find_largest_fuzzy_jump(l), find_largest_fuzzy_jump(c), find_largest_fuzzy_jump(r)
    if abs(hc - hl) < abs(hr - hc) and abs(hc - hl) < abs(hr - hl): a, b = hl, hc
    elif abs(hc - hl) > abs(hr - hc) and abs(hr - hc) < abs(hr - hl): a, b = hc, hr
    else: a, b = hl, hr
    return max(0, min(int((a + b) // 2), 192 - 10))

def run_fcm_corrected(features, num_clusters=4, m=2.0, maxiter=25, seed=42):
    np.random.seed(seed)
    N, Q = features.shape
    u = np.random.rand(num_clusters, N)
    u /= np.sum(u, axis=0, keepdims=True)
    
    for _ in range(maxiter):
        u_old = u.copy()
        um = u ** m
        um_sum = np.sum(um, axis=1, keepdims=True) + 1e-10
        centers = (um @ features) / um_sum
        diff = features[:, None, :] - centers[None, :, :]
        d = np.linalg.norm(diff, axis=2).T
        d = np.fmax(d, 1e-6)
        inv_d = d ** (-2.0 / (m - 1.0))
        u = inv_d / (np.sum(inv_d, axis=0, keepdims=True) + 1e-10)
        if np.linalg.norm(u - u_old) < 0.01:
            break
    return centers, u

def segment_fcm_pixel_level(water_bgr, num_clusters=4, min_contrast=0.20):
    """
    PASO 1: Detectar agua y obstáculos a nivel de píxel en la región de navegación.
    NO genera cajas aquí. Solo produce las máscaras semánticas.
    """
    H_w, W_w, _ = water_bgr.shape
    if H_w < 5:
        return np.zeros((H_w, W_w), dtype=np.uint8), np.zeros((H_w, W_w), dtype=np.uint8)
        
    # Filtrado suave para evitar que el oleaje rompa el fondo
    filtered = cv2.bilateralFilter(water_bgr, 5, 50, 50)
    
    # Supresión de degradado vertical (normalización de fondo lacustre)
    row_med = np.median(filtered, axis=1, keepdims=True)
    row_smooth = cv2.GaussianBlur(row_med.astype(np.float32), (1, 15), 0)
    res = filtered.astype(np.float32) - row_smooth + 128.0
    res_norm = np.clip(res, 0, 255) / 255.0
    
    data = res_norm.reshape(H_w * W_w, 3)
    cntr, u = run_fcm_corrected(data, num_clusters=num_clusters, m=2.0)
    labels = np.argmax(u, axis=0).reshape((H_w, W_w))
    
    # Centroide de agua: el más cercano a neutral 128 (0.5)
    dists_to_neutral = [np.linalg.norm(cntr[c_id] - 0.5) for c_id in range(num_clusters)]
    water_c_id = np.argmin(dists_to_neutral)
    
    # Máscara semántica de agua
    mask_agua = np.zeros((H_w, W_w), dtype=np.uint8)
    mask_agua[labels == water_c_id] = 255
    
    # Máscara semántica de obstáculos (clusters con contraste significativo)
    mask_obstaculos = np.zeros((H_w, W_w), dtype=np.uint8)
    for c_id in range(num_clusters):
        if c_id == water_c_id:
            continue
        contrast = np.linalg.norm(cntr[c_id] - cntr[water_c_id])
        if contrast >= min_contrast:
            mask_obstaculos[labels == c_id] = 255
            
    # Limpieza morfológica para unir regiones compactas y suprimir ruido aislado
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_obstaculos = cv2.morphologyEx(mask_obstaculos, cv2.MORPH_OPEN, kernel)
    
    return mask_agua, mask_obstaculos

def extract_boxes_from_mask(mask_obs, y_offset, min_area=35):
    """
    PASO 2: Aplicar algoritmo de encuadre sobre la máscara de obstáculos resultante.
    Compensa verticalmente las coordenadas con y_offset.
    """
    raw_boxes = mask_to_bounding_boxes(mask_obs, min_area=min_area)
    compensated_boxes = []
    for b in raw_boxes:
        c = dict(b)
        c["y_init"] = int(b["y_init"] + y_offset)
        c["y_end"] = int(b["y_end"] + y_offset)
        c["y_centroid"] = int(b["y_centroid"] + y_offset)
        c["_compensated"] = True
        compensated_boxes.append(c)
    return compensated_boxes

def evaluate_pipeline_on_approved_frames():
    print(f"[+] Evaluando pipeline nuevo sobre los {len(approved_frames)} frames aprobados por el usuario...")
    
    cap = cv2.VideoCapture(str(project_root / "assets/videos/tesis.mp4"))
    
    tp, fp, fn, tn = 0, 0, 0, 0
    results_per_frame = {}
    
    for idx, f_idx in enumerate(approved_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        img_256 = cv2.resize(frame, (256, 192))
        y_h = get_horizon(img_256)
        
        # Ground Truth del frame:
        gt_has_obs = bool((0 <= f_idx <= 900) or (1160 <= f_idx <= 1760))
        
        # 1. Segmentar agua y obstáculo en FCM (A nivel de pixel, sin encuadrar)
        water_bgr = img_256[y_h:, :]
        m_agua, m_obs = segment_fcm_pixel_level(water_bgr, num_clusters=4, min_contrast=0.20)
        
        # 2. Aplicar algoritmo de encuadre sobre la máscara
        boxes_fcm = extract_boxes_from_mask(m_obs, y_offset=y_h, min_area=35)
        
        # 3. Detector RGB
        _, boxes_rgb_raw = detector_rgb_cpu(water_bgr)
        boxes_rgb = []
        for b in boxes_rgb_raw:
            c = dict(b)
            c["y_init"] = int(b["y_init"] + y_h)
            c["y_end"] = int(b["y_end"] + y_h)
            c["y_centroid"] = int(b["y_centroid"] + y_h)
            c["_compensated"] = True
            boxes_rgb.append(c)
            
        # 4. Fusión difusa y 5. Confirmación por INTERSECCIÓN (sin memoria temporal)
        res_union = fuzzy_union([boxes_rgb, boxes_fcm])
        confirmed = intersect_fuzzy_detections(boxes_fcm, boxes_rgb, tol=15)
        
        det = len(confirmed) > 0
        if gt_has_obs and det:
            tp += 1
            cl = "TP"
        elif not gt_has_obs and det:
            fp += 1
            cl = "FP"
        elif gt_has_obs and not det:
            fn += 1
            cl = "FN"
        else:
            tn += 1
            cl = "TN"
            
        results_per_frame[f_idx] = {
            "gt": gt_has_obs,
            "detected": det,
            "classification": cl,
            "num_boxes_fcm": len(boxes_fcm),
            "num_boxes_confirmed": len(confirmed)
        }
        
        if (idx + 1) % 25 == 0 or idx == len(approved_frames) - 1:
            print(f"  Progreso: {idx+1}/{len(approved_frames)} frames | TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            
    cap.release()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("  RESULTADOS DEL PIPELINE DIFUSO CON NUEVA ARQUITECTURA")
    print("=" * 60)
    print(f"Frames aprobados evaluados: {len(approved_frames)}")
    print(f"TP: {tp:3d}  |  FP: {fp:3d}")
    print(f"FN: {fn:3d}  |  TN: {tn:3d}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_pipeline_on_approved_frames()
