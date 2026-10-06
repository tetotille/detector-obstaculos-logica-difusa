import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "WaSR-T"))
sys.path.append(str(project_root))

import cv2
import torch
import numpy as np
from PIL import Image
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights
from wasr_t.data.transforms import PytorchHubNormalization
from src.utils.utils import mask_to_bounding_boxes
from src.detector_horizonte.pixel_detector import separate_pixels, find_largest_fuzzy_jump

CACHE_DIR = project_root / "main_output/wasrt_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Frames de entrenamiento/calibración representativos:
# Frames con obstáculo: 50, 100, 200, 300, 400, 500, 600, 700, 800, 1200, 1300, 1400, 1500, 1600
# Frames agua limpia: 920, 960, 1000, 1040, 1080, 1120, 1780
TUNING_FRAMES = [
    50, 100, 200, 300, 400, 500, 600, 700, 800, 1200, 1300, 1400, 1500, 1600,
    920, 960, 1000, 1040, 1080, 1120, 1780
]

def get_horizon(img_256):
    left, center, right = separate_pixels(img_256)
    h_l = find_largest_fuzzy_jump(left)
    h_c = find_largest_fuzzy_jump(center)
    h_r = find_largest_fuzzy_jump(right)
    if abs(h_c - h_l) < abs(h_r - h_c) and abs(h_c - h_l) < abs(h_r - h_l):
        a, b = h_l, h_c
    elif abs(h_c - h_l) > abs(h_r - h_c) and abs(h_r - h_c) < abs(h_r - h_l):
        a, b = h_c, h_r
    else:
        a, b = h_l, h_r
    ajuste = int((a + b) // 2)
    return max(0, min(ajuste, 192 - 10))

def extract_and_cache():
    print(f"[+] Inicializando WaSR-T para extraer pseudo-Ground Truth sobre {len(TUNING_FRAMES)} frames...")
    device = torch.device('cpu')
    model = wasr_temporal_resnet101(pretrained=False, hist_len=5)
    model.load_state_dict(load_weights(project_root / "WaSR-T/wasrt_mastr1478.pth"))
    model = model.sequential().eval().to(device)
    transform = PytorchHubNormalization()

    cap = cv2.VideoCapture(str(project_root / "assets/videos/tesis.mp4"))
    
    for f_idx in TUNING_FRAMES:
        cache_file = CACHE_DIR / f"frame_{f_idx:04d}.npz"
        if cache_file.exists():
            print(f"  [-] Frame {f_idx} ya en cache.")
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  [!] Error leyendo frame {f_idx}")
            continue

        img_256 = cv2.resize(frame, (256, 192))
        y_horizon = get_horizon(img_256)

        # Inferencia WaSR-T a 512x384
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512, 384))
        inp = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            res = model({'image': inp})
            
        probs = res['out'][0].cpu().numpy()
        pred_512 = probs.argmax(0).astype(np.uint8) # 0=obs/shore, 1=water, 2=sky
        
        # Redimensionar predicción a 256x192
        pred_256 = cv2.resize(pred_512, (256, 192), interpolation=cv2.INTER_NEAREST)
        
        # Máscara de obstáculo en agua: clase 0 por debajo del horizonte
        obs_water_mask = ((pred_256 == 0) & (np.arange(192)[:, None] >= y_horizon)).astype(np.uint8)
        water_mask = ((pred_256 == 1) & (np.arange(192)[:, None] >= y_horizon)).astype(np.uint8)
        
        np.savez_compressed(
            str(cache_file),
            frame_idx=f_idx,
            y_horizon=y_horizon,
            pred_256=pred_256,
            obs_water_mask=obs_water_mask,
            water_mask=water_mask,
            img_256=img_256
        )
        print(f"  [+] Frame {f_idx} procesado y guardado en cache (Horizonte y={y_horizon}, Obs px={np.sum(obs_water_mask)})")
        
    cap.release()
    print("[+] Extracción completada.")

if __name__ == "__main__":
    extract_and_cache()
