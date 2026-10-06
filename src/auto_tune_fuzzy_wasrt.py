import numpy as np
import cv2
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
CACHE_DIR = project_root / "main_output/wasrt_cache"

# Cargar los 21 frames cacheados
def load_all_cached_data():
    cache_files = sorted(CACHE_DIR.glob("frame_*.npz"))
    dataset = []
    for cf in cache_files:
        d = np.load(cf)
        dataset.append({
            "frame_idx": int(d["frame_idx"]),
            "y_horizon": int(d["y_horizon"]),
            "img_256": d["img_256"],
            "obs_water_mask": d["obs_water_mask"],
            "water_mask": d["water_mask"]
        })
    return dataset

# Implementación estándar vectorizada de FCM
def run_fcm_corrected(features, num_clusters=3, m=2.0, maxiter=20, error=0.01, seed=42):
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
        
        if np.linalg.norm(u - u_old) < error:
            break
            
    return centers, u

def segment_water_frame(img_256, y_horizon, config):
    """
    Segmenta los obstáculos en el agua usando la configuración dada.
    """
    water = img_256[y_horizon:, :]
    H_w, W_w, _ = water.shape
    if H_w < 5 or W_w < 5:
        return np.zeros((192, 256), dtype=np.uint8)
        
    # 1. Filtro bilateral opcional
    if config.get("use_bilateral", False):
        d_val = config.get("bilateral_d", 5)
        proc_water = cv2.bilateralFilter(water, d_val, 50, 50)
    else:
        proc_water = water.copy()
        
    # 2. Supresión de degradado vertical (Detrending)
    if config.get("detrend_gradient", True):
        ksize = config.get("detrend_ksize", 15)
        row_median = np.median(proc_water, axis=1, keepdims=True)
        row_smooth = cv2.GaussianBlur(row_median.astype(np.float32), (1, ksize), 0)
        diff = proc_water.astype(np.float32) - row_smooth + 128.0
        proc_water_norm = np.clip(diff, 0, 255).astype(np.uint8)
    else:
        proc_water_norm = proc_water
        
    # 3. Espacio de color
    color_space = config.get("color_space", "BGR")
    if color_space == "LAB":
        feat_img = cv2.cvtColor(proc_water_norm, cv2.COLOR_BGR2LAB)
    elif color_space == "HSV":
        feat_img = cv2.cvtColor(proc_water_norm, cv2.COLOR_BGR2HSV)
    else:
        feat_img = proc_water_norm
        
    features = (feat_img.astype(np.float32) / 255.0).reshape(H_w * W_w, 3)
    
    # 4. FCM
    num_c = config.get("num_clusters", 3)
    m_val = config.get("m", 2.0)
    cntr, u = run_fcm_corrected(features, num_clusters=num_c, m=m_val, maxiter=20, seed=42)
    labels = np.argmax(u, axis=0).reshape((H_w, W_w))
    
    # 5. Identificar el cluster dominante de agua (el más cercano al residuo neutro 128/255=0.5 o con mayor área)
    if config.get("detrend_gradient", True):
        dists_to_neutral = [np.linalg.norm(cntr[c_id] - 0.5) for c_id in range(num_c)]
        water_c_id = np.argmin(dists_to_neutral)
    else:
        # Cluster con mayor número de píxeles
        counts = [np.sum(labels == c_id) for c_id in range(num_c)]
        water_c_id = np.argmax(counts)
        
    # 6. Identificar candidatos a obstáculo
    # Un cluster es obstáculo si su distancia/contraste con el cluster de agua supera min_contrast
    min_contrast = config.get("min_contrast", 0.12)
    obs_water_pred = np.zeros((H_w, W_w), dtype=np.uint8)
    
    for c_id in range(num_c):
        if c_id == water_c_id:
            continue
        contrast = np.linalg.norm(cntr[c_id] - cntr[water_c_id])
        if contrast >= min_contrast:
            # Píxeles pertenecientes a este cluster
            c_mask = (labels == c_id).astype(np.uint8)
            # Filtro morfológico o componente conexa mínima
            min_area = config.get("min_area", 25)
            num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(c_mask)
            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area >= min_area:
                    obs_water_pred[comp_labels == comp_id] = 1
                    
    # Reconstruir a tamaño completo 256x192
    full_mask = np.zeros((192, 256), dtype=np.uint8)
    full_mask[y_horizon:, :] = obs_water_pred
    return full_mask

def evaluate_config(dataset, config):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious = []
    
    for sample in dataset:
        gt_obs = sample["obs_water_mask"]
        pred_obs = segment_water_frame(sample["img_256"], sample["y_horizon"], config)
        
        intersection = np.sum((gt_obs == 1) & (pred_obs == 1))
        union = np.sum((gt_obs == 1) | (pred_obs == 1))
        
        tp = intersection
        fp = np.sum((gt_obs == 0) & (pred_obs == 1))
        fn = np.sum((gt_obs == 1) & (pred_obs == 0))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if union > 0:
            ious.append(intersection / union)
        else:
            # Ambos vacíos -> predicción perfecta de agua limpia
            ious.append(1.0)
            
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou = float(np.mean(ious))
    
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mean_iou": mean_iou
    }

def main():
    dataset = load_all_cached_data()
    print(f"[+] Datos de calibración cargados: {len(dataset)} frames con máscaras WaSR-T.")
    
    # Espacio de búsqueda de parámetros
    color_spaces = ["BGR", "LAB"]
    use_bilaterals = [False, True]
    detrends = [True, False]
    num_clusters_list = [3, 4]
    min_contrasts = [0.08, 0.12, 0.16, 0.20]
    min_areas = [20, 35, 50]
    
    best_f1 = -1.0
    best_config = None
    best_metrics = None
    
    results = []
    
    # Búsqueda estructurada
    print("\n[+] Iniciando optimización automática de parámetros frente a WaSR-T...")
    count = 0
    for cs in color_spaces:
        for detrend in [True]: # Detrending es clave por lo observado
            for ub in [False, True]:
                for nc in [3, 4]:
                    for mc in [0.08, 0.12, 0.15, 0.20]:
                        for ma in [25, 40]:
                            cfg = {
                                "color_space": cs,
                                "use_bilateral": ub,
                                "bilateral_d": 5,
                                "detrend_gradient": detrend,
                                "detrend_ksize": 15,
                                "num_clusters": nc,
                                "m": 2.0,
                                "min_contrast": mc,
                                "min_area": ma
                            }
                            met = evaluate_config(dataset, cfg)
                            results.append({"config": cfg, "metrics": met})
                            count += 1
                            if met["f1"] > best_f1:
                                best_f1 = met["f1"]
                                best_config = cfg
                                best_metrics = met
                                print(f"  [Iter {count:3d}] NUEVO MEJOR F1: {best_f1:.4f} (IoU={met['mean_iou']:.4f}, Prec={met['precision']:.4f}, Rec={met['recall']:.4f}) | CS={cs}, Bilat={ub}, C={nc}, Contrast={mc}, Area={ma}")
                                
    print("\n" + "=" * 65)
    print("  OPTIMIZACIÓN COMPLETADA CON ÉXITO")
    print("=" * 65)
    print("Mejores métricas obtenidas frente a WaSR-T:")
    print(f"  - F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  - Mean IoU:  {best_metrics['mean_iou']:.4f}")
    print(f"  - Precisión: {best_metrics['precision']:.4f}")
    print(f"  - Recall:    {best_metrics['recall']:.4f}")
    print("\nMejor configuración óptima:")
    print(json.dumps(best_config, indent=2))
    
    # Guardar resultados
    out_file = project_root / "main_output/fcm_calibracion_wasrt.json"
    with open(out_file, "w") as f:
        json.dump({"best_config": best_config, "best_metrics": best_metrics}, f, indent=2)
    print(f"\n[+] Resultados guardados en: {out_file}")

if __name__ == "__main__":
    main()
