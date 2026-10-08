import cv2
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.utils import read_image
from src.detector_horizonte.pixel_detector import separate_pixels, find_largest_fuzzy_jump
from src.cmeans.c_means_main import fcm, cmeans

OUTPUT_DIR = project_root / "main_output/fcm_diagnostico"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def diagnostico_frame(frame_idx: int, frame_bgr: np.ndarray):
    w, h = 256, 192
    img_np, _ = read_image(frame_bgr, w, h)
    
    # 1. Detección horizonte
    left, center, right = separate_pixels(img_np)
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
    
    # 2. Ejecutar FCM actual
    mask_actual, _, cuadros_actual = fcm(img_np, num_clusters=4, punto_horizonte=ajuste)
    
    # 3. Analizar cómo se comportan los clusters en FCM actual
    # Notar que fcm() toma la imagen completa, la normaliza y llama a cmeans
    img_float = (img_np.astype(np.float32) / 255.0)
    S, N = img_float.shape[0] * img_float.shape[1], img_float.shape[2]
    data = img_float.reshape(S, N)
    
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, 4, 2.0, error=0.05, maxiter=10)
    cluster_labels = np.argmax(u, axis=0).reshape((h, w))
    
    # Recorte bajo horizonte
    water_img = img_np[ajuste:, :]
    water_clusters = cluster_labels[ajuste:, :]
    
    # Colores visuales para los 4 clusters
    cluster_colors = np.array([
        [255, 0, 0],    # Azul (BGR)
        [0, 255, 0],    # Verde
        [0, 0, 255],    # Rojo
        [0, 255, 255]   # Amarillo
    ], dtype=np.uint8)
    
    vis_clusters_full = cluster_colors[cluster_labels]
    vis_clusters_water = cluster_colors[water_clusters]
    
    # Conteo de píxeles por cluster bajo el agua
    total_water_px = water_clusters.size
    print(f"\n--- Frame {frame_idx} (Horizonte y={ajuste}, Total px agua={total_water_px}) ---")
    print(f"Iteraciones convergencia FCM: {p}, FPC: {fpc:.4f}")
    
    cluster_stats = []
    for c_id in range(4):
        count = int(np.sum(water_clusters == c_id))
        pct = (count / total_water_px) * 100
        center_bgr = (cntr[c_id] * 255.0).astype(int)
        cluster_stats.append((c_id, count, pct, center_bgr))
        print(f"  Cluster {c_id}: {count:5d} px ({pct:5.1f}%) | Centro BGR: {center_bgr.tolist()}")
    
    # Ver qué elige argmin(counts)
    counts_in_water = [int(np.sum(water_clusters == c_id)) for c_id in range(4)]
    # Filtrar clusters que tengan al menos 1 pixel
    valid_counts = [(c_id, cnt) for c_id, cnt in enumerate(counts_in_water) if cnt > 0]
    argmin_choice = min(valid_counts, key=lambda x: x[1])[0]
    print(f"  --> FCM actual (argmin) eligió Cluster {argmin_choice} con {counts_in_water[argmin_choice]} px")
    print(f"  --> Cajas detectadas por FCM actual: {len(cuadros_actual)}")
    
    # Generar visualización combinada
    # Fila 1: Imagen original redimensionada con horizonte | Mapa de 4 clusters completo | Mapa de 4 clusters en agua
    row1_p1 = img_np.copy()
    cv2.line(row1_p1, (0, ajuste), (w, ajuste), (255, 0, 0), 2)
    cv2.putText(row1_p1, f"Frame {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    row1_p2 = vis_clusters_full.copy()
    cv2.line(row1_p2, (0, ajuste), (w, ajuste), (255, 255, 255), 1)
    
    row1_p3 = np.zeros_like(img_np)
    row1_p3[ajuste:, :] = vis_clusters_water
    
    row1 = np.hstack([row1_p1, row1_p2, row1_p3])
    
    # Fila 2: Cada uno de los 4 clusters de agua aislados
    panels_c = []
    for c_id in range(4):
        mask_c = np.zeros_like(img_np)
        mask_c[ajuste:, :] = (water_clusters == c_id).astype(np.uint8)[:, :, None] * 255
        # Añadir texto indicando si fue el elegido por argmin
        is_chosen = "(ELEGIDO)" if c_id == argmin_choice else ""
        cv2.putText(mask_c, f"C{c_id}: {counts_in_water[c_id]}px {is_chosen}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if is_chosen else (200, 200, 200), 1)
        panels_c.append(mask_c)
    
    # Combinar en una imagen diagnóstica
    row2 = np.hstack(panels_c) # Ancho: 256*4 = 1024
    row1_resized = cv2.resize(row1, (1024, int(192 * (1024 / (256 * 3)))))
    
    diagnostic_img = np.vstack([row1_resized, row2])
    out_path = OUTPUT_DIR / f"diagnostico_frame_{frame_idx:04d}.png"
    cv2.imwrite(str(out_path), diagnostic_img)
    print(f"  [+] Guardado: {out_path}")
    return out_path

def main():
    cap = cv2.VideoCapture(str(project_root / "assets/videos/tesis.mp4"))
    test_frames = [50, 100, 200, 500, 1000, 1400]
    
    for f_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            diagnostico_frame(f_idx, frame)
    cap.release()
    print("\nDiagnóstico completado para todos los frames de prueba.")

if __name__ == "__main__":
    main()
