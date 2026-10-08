import numpy as np
import cv2
import sys
from os.path import join, dirname, abspath
project_root = dirname(dirname(dirname(abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scipy.ndimage import label
from src.utils.utils import neighbor_framed_np, mask_to_bounding_boxes


def calculate_distances(data, centers):
    """
    Calcula la distancia euclidiana L2 entre cada punto de datos (N, Q)
    y cada centroide (C, Q).
    Retorna matriz de distancias con forma (C, N).
    """
    diff = data[:, None, :] - centers[None, :, :]  # (N, C, Q)
    dists = np.linalg.norm(diff, axis=2).T         # (C, N)
    return np.fmax(dists, 1e-6)

def cmeans(data, c, m=2.0, error=0.01, maxiter=25, metric='euclidean', init=None, seed=42):
    """
    Algoritmo Fuzzy C-Means estándar (Bezdek 1981).
    Fórmula exacta:
      u_ik = 1 / sum_j (d_ik / d_jk)**(2 / (m - 1))
      c_i  = sum_k (u_ik**m * x_k) / sum_k (u_ik**m)
    
    data: (N_features, N_samples) o (N_samples, N_features)
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Asegurar formato (N_samples, N_features)
    if data.shape[0] < data.shape[1] and data.shape[0] in [1, 2, 3]:
        data_pts = data.T
    else:
        data_pts = data
        
    N, Q = data_pts.shape
    
    if init is None:
        u = np.random.rand(c, N)
        u /= np.sum(u, axis=0, keepdims=True)
    else:
        u = np.array(init, dtype=np.float32)
        u /= np.sum(u, axis=0, keepdims=True)
        
    jm = []
    p = 0
    
    for p in range(maxiter):
        u_old = u.copy()
        um = u ** m
        um_sum = np.sum(um, axis=1, keepdims=True) + 1e-10
        cntr = (um @ data_pts) / um_sum
        
        d = calculate_distances(data_pts, cntr)
        
        # Función objetivo J_m = sum_i sum_k u_ik^m * d_ik^2
        jm_val = np.sum(um * (d ** 2))
        jm.append(jm_val)
        
        # Actualización de matriz de pertenencia difusa: d ** (-2 / (m - 1))
        inv_d = d ** (-2.0 / (m - 1.0))
        u = inv_d / (np.sum(inv_d, axis=0, keepdims=True) + 1e-10)
        
        if np.linalg.norm(u - u_old) < error:
            break
            
    # Coeficiente de partición difusa (FPC)
    fpc = float(np.trace(u @ u.T) / float(N))
    
    return cntr, u, None, d, np.array(jm), p, fpc

def segment_fcm_pixel_level(water_bgr, num_clusters=4, m=2.0, min_contrast=0.20):
    """
    PASO 1: Segmenta el agua y los obstáculos a nivel de píxel/cluster
    sobre la región navegable debajo del horizonte, SIN encuadrar primero.
    
    Parámetros:
    - water_bgr: Imagen recortada en la zona navegable (H_w, W_w, 3)
    - num_clusters: Número de clusters difusos (por defecto 4)
    - m: Exponente de borrosidad (por defecto 2.0)
    - min_contrast: Umbral mínimo de contraste residual frente al cluster de agua
    
    Retorna:
    - mask_agua: Máscara binaria (uint8 [0, 255]) del agua limpia
    - mask_obstaculos: Máscara binaria (uint8 [0, 255]) de los obstáculos detectados
    - info: Diccionario con centroides y cluster de agua
    """
    H_w, W_w, _ = water_bgr.shape
    if H_w < 5 or W_w < 5:
        empty = np.zeros((H_w, W_w), dtype=np.uint8)
        return empty, empty, {"water_c_id": 0, "contrast_max": 0.0}
        
    # 1. Filtrado bilateral para amortiguar texturas de oleaje
    filtered = cv2.bilateralFilter(water_bgr, 5, 50, 50)
    
    # 2. Supresión del degradado vertical de luminosidad (normalización del fondo)
    row_med = np.median(filtered, axis=1, keepdims=True)
    row_smooth = cv2.GaussianBlur(row_med.astype(np.float32), (1, 15), 0)
    res = filtered.astype(np.float32) - row_smooth + 128.0
    res_norm = np.clip(res, 0, 255) / 255.0
    
    # 3. Clustering FCM en espacio residual
    data = res_norm.reshape(H_w * W_w, 3)
    cntr, u, _, _, _, _, _ = cmeans(data, c=num_clusters, m=m, error=0.01, maxiter=25, seed=42)
    labels = np.argmax(u, axis=0).reshape((H_w, W_w))
    counts = [int(np.sum(labels == c_id)) for c_id in range(num_clusters)]
    
    # 4. Identificar cluster dominante de agua (más cercano a 0.5 residual)
    dists_to_neutral = [float(np.linalg.norm(cntr[c_id] - 0.5)) for c_id in range(num_clusters)]
    water_c_id = int(np.argmin(dists_to_neutral))
    
    # 5. Cluster minoritario candidato a obstáculo
    min_c_id = int(np.argmin(counts))
    contrast_min = float(np.linalg.norm(cntr[min_c_id] - cntr[water_c_id]))
    
    mask_agua = np.zeros((H_w, W_w), dtype=np.uint8)
    mask_obstaculos = np.zeros((H_w, W_w), dtype=np.uint8)
    
    # Un cluster solo es obstáculo si tiene contraste residual significativo y ocupa < 25% del agua
    if contrast_min >= min_contrast and counts[min_c_id] < (H_w * W_w * 0.25):
        mask_obstaculos[labels == min_c_id] = 255
        mask_agua[labels != min_c_id] = 255
    else:
        # Agua 100% limpia sin obstáculos
        mask_agua[:, :] = 255
        
    # Limpieza morfológica para suprimir píxeles aislados de oleaje
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_obstaculos = cv2.morphologyEx(mask_obstaculos, cv2.MORPH_OPEN, kernel)
    
    info = {
        "water_c_id": water_c_id,
        "obstacle_c_id": min_c_id,
        "contrast": contrast_min,
        "centroids": cntr.tolist()
    }
    return mask_agua, mask_obstaculos, info

def is_valid_obstacle_geometry(width, height, max_width):
    """
    Filtro geométrico discriminador: diferencia cuerpos 2D compactos de crestas de oleaje planas.
    Las crestas de oleaje son tiras horizontales finas (h < 8 px o w/h > 2.8).
    Los obstáculos reales (boyas, lanchas, pelotas) tienen altura h >= 8 px y compacidad 2D.
    """
    if width >= max_width * 0.85:
        return False
    if height >= 12:
        return (width / height) <= 6.5
    elif height >= 8:
        return (width / height) <= 2.8
    return False

def extract_boxes_from_mask(mask_obstaculos, y_offset=0, min_area=25):
    """
    PASO 2: Aplica el algoritmo de encuadre sobre la máscara binaria limpia de obstáculos.
    Filtra estelas de oleaje plano y compensa verticalmente sumando y_offset.
    """
    H, W = mask_obstaculos.shape
    raw_boxes = mask_to_bounding_boxes(mask_obstaculos, min_area=min_area)
    compensated_boxes = []
    for b in raw_boxes:
        w = b["x_end"] - b["x_init"]
        h = b["y_end"] - b["y_init"]
        if is_valid_obstacle_geometry(w, h, W):
            c = dict(b)
            c["y_init"] = int(b["y_init"] + y_offset)
            c["y_end"] = int(b["y_end"] + y_offset)
            c["y_centroid"] = int(b["y_centroid"] + y_offset)
            c["_compensated"] = True
            compensated_boxes.append(c)
    return compensated_boxes

def fcm(resized_image, num_clusters=4, m=2.0, metric='euclidean', show_images=False, punto_horizonte=0):
    """
    Función principal de FCM para compatibilidad retroactiva.
    1. Segmenta agua y obstáculos a nivel de píxel.
    2. Aplica el algoritmo de encuadre posteriormente sobre la máscara de obstáculos.
    """
    H, W, _ = resized_image.shape
    y_nav_start = max(0, min(punto_horizonte + 5, H - 20))
    y_nav_end = min(174, H)
    water_bgr = resized_image[y_nav_start:y_nav_end, :]
    
    # 1. Segmentar sin encuadrar primero
    mask_agua, mask_obstaculos_roi, _ = segment_fcm_pixel_level(
        water_bgr, num_clusters=num_clusters, m=m, min_contrast=0.245
    )
    
    # Máscara completa referenciada a la imagen completa
    mask_obstaculos = np.zeros((H, W), dtype=np.uint8)
    mask_obstaculos[y_nav_start:y_nav_end, :] = mask_obstaculos_roi
    
    # 2. Aplicar algoritmo de encuadre sobre la máscara resultante
    cuadros = extract_boxes_from_mask(mask_obstaculos_roi, y_offset=y_nav_start, min_area=25)
    
    return mask_obstaculos, punto_horizonte, cuadros

def fcm_semantic(resized_image, num_clusters=4, m=2.0, punto_horizonte=0):
    """
    Variante extendida que retorna ambas máscaras (agua y obstáculos) y las cajas.
    """
    H, W, _ = resized_image.shape
    y_nav_start = max(0, min(punto_horizonte + 5, H - 20))
    y_nav_end = min(174, H)
    water_bgr = resized_image[y_nav_start:y_nav_end, :]
    
    mask_agua_roi, mask_obstaculos_roi, info = segment_fcm_pixel_level(
        water_bgr, num_clusters=num_clusters, m=m, min_contrast=0.245
    )
    
    mask_agua = np.zeros((H, W), dtype=np.uint8)
    mask_agua[y_nav_start:y_nav_end, :] = mask_agua_roi
    mask_obstaculos = np.zeros((H, W), dtype=np.uint8)
    mask_obstaculos[y_nav_start:y_nav_end, :] = mask_obstaculos_roi
    
    cuadros = extract_boxes_from_mask(mask_obstaculos_roi, y_offset=y_nav_start, min_area=25)
    
    return mask_agua, mask_obstaculos, punto_horizonte, cuadros, info

if __name__ == "__main__":
    filename = join(dirname(dirname(dirname(abspath(__file__)))), "assets/images/barco.jpg")
    image = cv2.imread(filename)
    if image is not None:
        resized_image = cv2.resize(image, (256, 192))
        mask_obs, h_pt, cuadros = fcm(resized_image, num_clusters=4, punto_horizonte=60)
        print("FCM prueba barco.jpg:")
        print(f"  Píxeles obstáculo: {np.sum(mask_obs > 0)}")
        print(f"  Cajas encontradas: {len(cuadros)}")
        for c in cuadros:
            print(f"    Caja: x=[{c['x_init']}..{c['x_end']}], y=[{c['y_init']}..{c['y_end']}], peso={c['weight']}")
