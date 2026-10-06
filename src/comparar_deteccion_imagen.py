"""
Obstacle Detection Comparison: Original vs Proposed Fuzzy Logic vs WaSR-T CNN
Extracts:
1. Original input image.
2. Proposed Fuzzy Logic pipeline with blue horizon line and red obstacle bounding boxes.
3. WaSR-T Temporal ResNet-101 semantic segmentation with obstacle bounding boxes.
4. Horizontal 3-panel comparative grid.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from PIL import Image

# Add project root and WaSR-T directories to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "WaSR-T"))

from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights
from src.utils.utils import mask_to_bounding_boxes, read_image
from src.cmeans import fcm
from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv.rgb_detection import process_image_cpu as detector_rgb
from src.fuzzy_union.fuzzy_union import fuzzy_union

# Segmentation palette for WaSR-T
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],   # Obstacle (RGB) -> Yellow
    [41, 167, 224],   # Water (RGB) -> Blue
    [90, 75, 164]     # Sky (RGB) -> Purple/Gray
], np.uint8)

_wasrt_model = None
_wasrt_device = None

def get_wasrt_model(weights_path: Optional[Union[str, Path]] = None):
    """Loads and caches the WaSR-T model for fast subsequent inferences."""
    global _wasrt_model, _wasrt_device
    if _wasrt_model is not None:
        return _wasrt_model, _wasrt_device

    if weights_path is None:
        weights_path = project_root / "WaSR-T" / "wasrt_mastr1478.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = wasr_temporal_resnet101(pretrained=False, hist_len=5)
    state_dict = load_weights(str(weights_path))
    model.load_state_dict(state_dict)
    model = model.sequential().eval().to(device)
    model.clear_state()
    
    _wasrt_model = model
    _wasrt_device = device
    return _wasrt_model, _wasrt_device

def process_ours(image_bgr: np.ndarray, target_size: Tuple[int, int] = (512, 384)):
    """
    Executes the proposed fuzzy logic obstacle detection pipeline:
    1. Horizon line estimation
    2. Cropping below the horizon
    3. Mode-adaptive RGB & FCM detectors
    4. Fuzzy union fusion
    5. Rendering: Blue horizon line + Red obstacle bounding boxes ONLY.
    """
    t_w, t_h = target_size
    img_np, img_cp = read_image(image_bgr, 256, 192)
    
    # 1. Horizon estimation
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
    adjustment = int((a + b) // 2)
    
    # 2. Crop below horizon
    cropped_np = img_np[adjustment:, :]
    cropped_cp = img_cp[adjustment:, :]
    
    # 3. Detectors
    _, boxes_rgb = detector_rgb(cropped_cp)
    _, _, boxes_cmeans = fcm(img_np, 4, punto_horizonte=adjustment)
    
    # Adjust coordinates relative to the full image (256x192)
    for c in boxes_rgb:
        c["y_init"] += adjustment
        c["y_end"] += adjustment
        c["y_centroid"] += adjustment
        
    for c in boxes_cmeans:
        c["y_init"] += adjustment
        c["y_end"] += adjustment
        c["y_centroid"] += adjustment
        
    # 4. Fuzzy Union fusion
    copied_rgb = [c.copy() for c in boxes_rgb]
    copied_fcm = [c.copy() for c in boxes_cmeans]
    final_boxes = fuzzy_union([copied_rgb, copied_fcm])
    
    # 5. Render onto target_size image
    scale_x = t_w / 256.0
    scale_y = t_h / 192.0
    output_img = cv2.resize(image_bgr, (t_w, t_h))
    
    # Blue horizon line (BGR: (255, 0, 0))
    h_a = int(a * scale_y)
    h_b = int(b * scale_y)
    cv2.line(output_img, (0, h_a), (t_w // 2, h_a), (255, 0, 0), 2)
    cv2.line(output_img, (t_w // 2, h_b), (t_w, h_b), (255, 0, 0), 2)
    
    # Text with contrast outline
    cv2.putText(output_img, "Horizon", (15, max(20, h_a - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output_img, "Horizon", (15, max(20, h_a - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    # ONLY Red bounding boxes for confirmed obstacles (BGR: (0, 0, 255))
    for c in final_boxes:
        if c is None: 
            continue
        x1, y1 = int(c["x_init"] * scale_x), int(c["y_init"] * scale_y)
        x2, y2 = int(c["x_end"] * scale_x), int(c["y_end"] * scale_y)
        w_val = c.get("weight", 0)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label = f"Ours: {w_val}px"
        cv2.putText(output_img, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output_img, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    
    return output_img, final_boxes

def process_wasrt(image_bgr: np.ndarray, target_size: Tuple[int, int] = (512, 384)):
    """
    Executes the WaSR-T model on the input image:
    1. Preprocessing and inference
    2. Semantic segmentation mask
    3. Morphological bounding box framing for obstacles
    4. Rendering with translucent mask overlay + bounding boxes
    """
    t_w, t_h = target_size
    model, device = get_wasrt_model()
    transform = PytorchHubNormalization()
    
    # Prepare input tensor
    img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    img_resized = img_pil.resize(target_size, Image.BILINEAR)
    img_normalized = transform(img_resized)
    input_batch = {'image': img_normalized.unsqueeze(0).to(device)}
    
    # Inference
    model.clear_state()
    with torch.no_grad():
        res = model(input_batch)
        
    probs = res['out'].cpu().detach().numpy()
    pred_mask_idx = probs.argmax(1)[0].astype(np.uint8)
    pred_mask_idx = cv2.resize(pred_mask_idx, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Segmentation color overlay
    pred_mask_color = SEGMENTATION_COLORS[pred_mask_idx]
    res_bgr = cv2.cvtColor(pred_mask_color, cv2.COLOR_RGB2BGR)
    
    # Translucent blend with original image
    img_orig_resized = cv2.resize(image_bgr, target_size)
    blended = cv2.addWeighted(img_orig_resized, 0.65, res_bgr, 0.35, 0)
    
    # Morphological bounding boxes for obstacle class (index 0)
    obs_mask = (pred_mask_idx == 0).astype(np.uint8)
    boxes_wasrt = mask_to_bounding_boxes(obs_mask, min_area=30)
    
    # Draw WaSR-T bounding boxes in red
    for c in boxes_wasrt:
        x1, y1 = c["x_init"], c["y_init"]
        x2, y2 = c["x_end"], c["y_end"]
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label = f"WaSR-T: {c['weight']}px"
        cv2.putText(blended, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    
    return blended, boxes_wasrt

def compare_detection(
    image_input: Union[str, Path, np.ndarray],
    output_dir: Optional[str] = "main_output/comparativa_cuadros",
    target_size: Tuple[int, int] = (512, 384),
    show: bool = False
) -> Dict[str, Any]:
    """
    Main detection comparison function.
    
    Parameters:
    - image_input: File path (str/Path) or pre-loaded BGR image array (np.ndarray).
    - output_dir: Directory where individual images and composite comparison will be saved.
    - target_size: Output resolution tuple (width, height), default (512, 384).
    - show: If True, opens an interactive OpenCV display window.
    
    Returns:
    - Dictionary with processed images, detected bounding boxes, and saved file paths.
    """
    # 1. Load image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")
        img_bgr = cv2.imread(str(img_path))
        stem = img_path.stem
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input
        stem = "frame_custom"
    else:
        raise ValueError("image_input must be a file path or a NumPy array.")

    if img_bgr is None:
        raise ValueError("Failed to decode the input image.")

    t_w, t_h = target_size
    out_dir = Path(output_dir) if output_dir else Path("main_output/comparativa_cuadros")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Process all three views
    # View 1: Original
    img_orig = cv2.resize(img_bgr, (t_w, t_h))
    
    # View 2: Proposed Fuzzy Logic Algorithm
    img_ours, boxes_ours = process_ours(img_bgr, target_size=target_size)
    
    # View 3: WaSR-T with Bounding Box Framing
    img_wasrt, boxes_wasrt = process_wasrt(img_bgr, target_size=target_size)
    
    # 3. Add panel headers for comparison
    v_orig = img_orig.copy()
    v_ours = img_ours.copy()
    v_wasrt = img_wasrt.copy()
    
    def draw_header(img: np.ndarray, text: str, color: Tuple[int, int, int]):
        cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    draw_header(v_orig, " (a) Original Input", (255, 255, 255))
    draw_header(v_ours, " (b) Ours (Fuzzy Logic)", (0, 0, 255))
    draw_header(v_wasrt, " (c) WaSR-T (Temporal CNN)", (255, 100, 255))
    
    # Horizontal 3-panel composition
    combined = np.hstack([v_orig, v_ours, v_wasrt])
    
    # 4. Save individual and combined files
    path_orig = out_dir / f"{stem}_1_original.png"
    path_ours = out_dir / f"{stem}_2_ours_fuzzy.png"
    path_wasrt = out_dir / f"{stem}_3_wasrt.png"
    path_combined = out_dir / f"{stem}_comparativa_completa.png"
    
    cv2.imwrite(str(path_orig), img_orig)
    cv2.imwrite(str(path_ours), img_ours)
    cv2.imwrite(str(path_wasrt), img_wasrt)
    cv2.imwrite(str(path_combined), combined)
    
    print("\n" + "=" * 65)
    print(f"  DETECTION COMPARISON GENERATED: {stem}")
    print("=" * 65)
    print(f"  [1] Original saved to:      {path_orig}")
    print(f"  [2] Ours (Fuzzy) saved to:  {path_ours} (Boxes: {len([b for b in boxes_ours if b])})")
    print(f"  [3] WaSR-T saved to:        {path_wasrt} (Boxes: {len(boxes_wasrt)})")
    print(f"  [+] Complete grid saved to: {path_combined}")
    print("=" * 65 + "\n")
    
    if show and "DISPLAY" in os.environ:
        win_name = f"Comparison: {stem} | Original | Ours | WaSR-T"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1400, 420)
        cv2.imshow(win_name, combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return {
        "original": img_orig,
        "ours": img_ours,
        "wasrt": img_wasrt,
        "combined": combined,
        "paths": {
            "original": str(path_orig),
            "ours": str(path_ours),
            "wasrt": str(path_wasrt),
            "combined": str(path_combined)
        },
        "boxes_ours": boxes_ours,
        "boxes_wasrt": boxes_wasrt,
        # Spanish backward-compatibility aliases
        "cuadros_ours": boxes_ours,
        "cuadros_wasrt": boxes_wasrt
    }

# Backward-compatibility alias
comparar_deteccion = compare_detection

def main():
    parser = argparse.ArgumentParser(
        description="Compare obstacle detection: Original vs Ours (Fuzzy Logic) vs WaSR-T on an image."
    )
    parser.add_argument("image", type=str, help="Path to input image (e.g. assets/images/barco.jpg)")
    parser.add_argument("--output_dir", "-o", type=str, default="main_output/comparativa_cuadros", help="Output directory to save images")
    parser.add_argument("--width", type=int, default=512, help="Width of each panel (default: 512)")
    parser.add_argument("--height", type=int, default=384, help="Height of each panel (default: 384)")
    parser.add_argument("--show", action="store_true", help="Show interactive display window")
    
    args = parser.parse_args()
    compare_detection(
        args.image,
        output_dir=args.output_dir,
        target_size=(args.width, args.height),
        show=args.show
    )

if __name__ == "__main__":
    main()
