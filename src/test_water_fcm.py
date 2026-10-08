import cv2
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.utils import read_image
from src.detector_horizonte.pixel_detector import separate_pixels, find_largest_fuzzy_jump
from src.cmeans.c_means_main import cmeans

OUTPUT_DIR = Path("main_output/fcm_diagnostico")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_water_only_fcm(frame_idx: int, frame_bgr: np.ndarray):
    w, h = 256, 192
    img_np, _ = read_image(frame_bgr, w, h)
    
    # Horizonte
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
    
    water_img = img_np[ajuste:, :]
    H_w, W_w, _ = water_img.shape
    total_water_px = H_w * W_w
    
    # Normalizar agua
    water_float = water_img.astype(np.float32) / 255.0
    data_water = water_float.reshape(total_water_px, 3)
    
    # Ejecutar cmeans solo en agua con 3 clusters y con 4 clusters
    print(f"\n================ Frame {frame_idx} (y_horizon={ajuste}) ================")
    for num_c in [3, 4]:
        cntr, u, _, _, _, iters, fpc = cmeans(data_water.T, num_c, 2.0, error=0.05, maxiter=20, seed=42)
        labels = np.argmax(u, axis=0).reshape((H_w, W_w))
        
        print(f"\n-- FCM SOLO EN AGUA ({num_c} clusters) [iters={iters}, FPC={fpc:.3f}] --")
        for c_id in range(num_c):
            cnt = int(np.sum(labels == c_id))
            pct = cnt / total_water_px * 100
            bgr = (cntr[c_id] * 255.0).astype(int)
            print(f"   Cluster {c_id}: {cnt:5d} px ({pct:5.1f}%) | BGR: {bgr.tolist()}")
            
        # Guardar visualización
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype=np.uint8)[:num_c]
        vis_map = colors[labels]
        
        # Guardar panel
        cv2.imwrite(str(OUTPUT_DIR / f"water_fcm_c{num_c}_frame_{frame_idx:04d}.png"), vis_map)

def main():
    cap = cv2.VideoCapture(str(project_root / "assets/videos/tesis.mp4"))
    for f_idx in [50, 100, 200, 500, 1000]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            test_water_only_fcm(f_idx, frame)
    cap.release()

if __name__ == "__main__":
    main()
