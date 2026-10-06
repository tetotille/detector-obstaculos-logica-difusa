import os
import sys
import time
import cv2
import numpy as np
import psutil
import torch
from pathlib import Path
from PIL import Image

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "WaSR-T"))

from src.cmeans import fcm
from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv.rgb_detection import process_image_cpu as detector_ours
from src.utils.utils import read_image
from src.fuzzy_union.fuzzy_union import fuzzy_union

from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights
from wasr_t.data.transforms import PytorchHubNormalization

# Accuracy Colors (from manual masks)
# In OpenCV BGR: [35, 195, 249] is Yellow (Obstacle), [224, 167, 41] is Blue (Water), [164, 76, 90] is Purple (Sky)
OBSTACLE_COLOR_GT = [35, 195, 249]

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def calculate_metrics(pred_mask, gt_mask, name=""):
    # Flatten and find obstacle pixels
    # GT Obstacle
    gt_obs = np.all(gt_mask == OBSTACLE_COLOR_GT, axis=-1)
    
    # Pred Obstacle
    pred_obs = pred_mask > 0
    
    tp = np.logical_and(pred_obs, gt_obs).sum()
    fp = np.logical_and(pred_obs, np.logical_not(gt_obs)).sum()
    fn = np.logical_and(np.logical_not(pred_obs), gt_obs).sum()
    
    print(f"  [{name}] TP: {tp}, FP: {fp}, FN: {fn}")
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return precision, recall, f1

def benchmark_ours(video_path, images, manual_masks):
    print("\nBenchmarking 'Ours'...")
    
    # 1. Performance (Video)
    cap = cv2.VideoCapture(video_path)
    times = []
    max_mem = 0
    
    for i in range(50): # 50 frames
        ret, frame = cap.read()
        if not ret: break
        
        start = time.time()
        # Full pipeline
        img_np, _ = read_image(frame, 256, 192)
        left, center, right = separate_pixels(img_np)
        h_left, h_center, h_right = find_largest_fuzzy_jump(left), find_largest_fuzzy_jump(center), find_largest_fuzzy_jump(right)
        ajuste = (min(h_left, h_center, h_right) + max(h_left, h_center, h_right)) // 2
        
        cropped_img = img_np[ajuste:, :]
        _, cuadros_rgb = detector_ours(cropped_img)
        _, _, cuadros_cmeans = fcm(img_np, 4, punto_horizonte=ajuste)
        
        _ = fuzzy_union([cuadros_rgb, cuadros_cmeans])
        
        times.append(time.time() - start)
        max_mem = max(max_mem, get_memory())
    
    cap.release()
    fps = 1.0 / np.mean(times)
    
    # 2. Accuracy (Images)
    precisions, recalls, f1s = [], [], []
    for i, (img_p, gt_p) in enumerate(zip(images, manual_masks)):
        img = cv2.imread(img_p)
        gt = cv2.imread(gt_p)
        gt = cv2.resize(gt, (256, 192), interpolation=cv2.INTER_NEAREST)
        
        img_np, _ = read_image(img, 256, 192)
        # Re-implementing parts of process_image_cpu to get binary mask
        height, width, _ = img_np.shape
        # Calculate modes (simplified)
        b_mode = np.bincount(img_np[:, :, 0].flatten()).argmax()
        g_mode = np.bincount(img_np[:, :, 1].flatten()).argmax()
        r_mode = np.bincount(img_np[:, :, 2].flatten()).argmax()
        
        from src.detector_hsv.rgb_detection import red_intensity_vectorized, green_intensity_vectorized, blue_intensity_vectorized, classify_pixel_vectorized
        red_cat = red_intensity_vectorized(img_np[:, :, 2], r_mode)
        green_cat = green_intensity_vectorized(img_np[:, :, 1], g_mode)
        blue_cat = blue_intensity_vectorized(img_np[:, :, 0], b_mode)
        classifications = classify_pixel_vectorized(red_cat, blue_cat, green_cat)
        
        pred_obs = (classifications == 1).astype(np.uint8)
        
        p, r, f = calculate_metrics(pred_obs, gt, f"Ours Img {i}")
        precisions.append(p); recalls.append(r); f1s.append(f)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1s), fps, max_mem

def benchmark_wasrt(video_path, weights_path, images, manual_masks, use_cuda=True):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"\nBenchmarking WaSR-T (Device: {device})...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    model = wasr_temporal_resnet101(pretrained=False, hist_len=5)
    model.load_state_dict(load_weights(weights_path))
    model = model.sequential().eval().to(device)
    transform = PytorchHubNormalization()
    
    # 1. Performance (Video)
    cap = cv2.VideoCapture(video_path)
    times = []
    max_mem = 0
    
    for i in range(20): # 20 frames
        ret, frame = cap.read()
        if not ret: break
        
        start = time.time()
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((256, 192), Image.BILINEAR)
        img_normalized = transform(img_pil)
        input_batch = {'image': img_normalized.unsqueeze(0).to(device)}
        
        with torch.no_grad():
            res = model(input_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
        times.append(time.time() - start)
        max_mem = max(max_mem, get_memory())
        
    cap.release()
    fps = 1.0 / np.mean(times)
    max_gpu_mem = get_gpu_memory() if device.type == 'cuda' else 0.0
    
    # 2. Accuracy (Images)
    precisions, recalls, f1s = [], [], []
    for i, (img_p, gt_p) in enumerate(zip(images, manual_masks)):
        img = cv2.imread(img_p)
        gt = cv2.imread(gt_p)
        gt = cv2.resize(gt, (256, 192), interpolation=cv2.INTER_NEAREST)
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((256, 192), Image.BILINEAR)
        img_normalized = transform(img_pil)
        input_batch = {'image': img_normalized.unsqueeze(0).to(device)}
        
        model.clear_state()
        with torch.no_grad():
            res = model(input_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
        probs = res['out'].cpu().numpy()
        # In WaSR-T: Class 0 = Obstacle, Class 1 = Water, Class 2 = Sky
        pred_obs = (probs.argmax(1)[0] == 0).astype(np.uint8)
        pred_obs = cv2.resize(pred_obs, (256, 192), interpolation=cv2.INTER_NEAREST)
        
        p, r, f = calculate_metrics(pred_obs, gt, f"WaSR-T Img {i}")
        precisions.append(p); recalls.append(r); f1s.append(f)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1s), fps, max_mem, max_gpu_mem

if __name__ == "__main__":
    video = "assets/videos/tesis.mp4"
    weights = "WaSR-T/wasrt_mastr1478.pth"
    images = ["assets/images/akaso1.jpeg", "assets/images/akaso2.jpeg", "assets/images/akaso3.jpeg"]
    masks = ["WaSR-T/images/akaso1_manual.png", "WaSR-T/images/akaso2_manual.png", "WaSR-T/images/akaso3_manual.png"]
    
    # Run benchmarks
    try:
        p_ours, r_ours, f1_ours, fps_ours, mem_ours = benchmark_ours(video, images, masks)
        
        # Test WaSR-T on CUDA
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            p_wasr_gpu, r_wasr_gpu, f1_wasr_gpu, fps_wasr_gpu, mem_wasr_gpu, gpu_wasr = benchmark_wasrt(
                video, weights, images, masks, use_cuda=True
            )
        else:
            p_wasr_gpu, r_wasr_gpu, f1_wasr_gpu, fps_wasr_gpu, mem_wasr_gpu, gpu_wasr = (0, 0, 0, 0, 0, 0)

        # Test WaSR-T on CPU
        p_wasr_cpu, r_wasr_cpu, f1_wasr_cpu, fps_wasr_cpu, mem_wasr_cpu, _ = benchmark_wasrt(
            video, weights, images, masks, use_cuda=False
        )
        
        print("\n" + "="*86)
        print("                 COMPARATIVE BENCHMARK: WaSR-T vs FUZZY LOGIC")
        print("="*86)
        header = f"{'Metric':<20} | {'Ours (Fuzzy CPU)':<18} | {'WaSR-T (CPU)':<18}"
        if has_cuda:
            header += f" | {'WaSR-T (CUDA)':<18}"
        print(header)
        print("-" * len(header))
        
        def fmt_row(name, val_ours, val_w_cpu, val_w_gpu, fmt_spec=".4f"):
            row = f"{name:<20} | {val_ours:<18{fmt_spec}} | {val_w_cpu:<18{fmt_spec}}"
            if has_cuda:
                row += f" | {val_w_gpu:<18{fmt_spec}}"
            return row
            
        print(fmt_row("Precision", p_ours, p_wasr_cpu, p_wasr_gpu))
        print(fmt_row("Recall", r_ours, r_wasr_cpu, r_wasr_gpu))
        print(fmt_row("F1-Score", f1_ours, f1_wasr_cpu, f1_wasr_gpu))
        print(fmt_row("FPS (Throughput)", fps_ours, fps_wasr_cpu, fps_wasr_gpu, fmt_spec=".2f"))
        print(fmt_row("Latency (ms/frame)", 1000.0/fps_ours, 1000.0/fps_wasr_cpu, 1000.0/(fps_wasr_gpu if fps_wasr_gpu > 0 else 1), fmt_spec=".1f"))
        print(fmt_row("RAM Usage (MB)", mem_ours, mem_wasr_cpu, mem_wasr_gpu, fmt_spec=".1f"))
        if has_cuda:
            print(f"{'VRAM Usage (MB)':<20} | {'0.0':<18} | {'0.0':<18} | {gpu_wasr:<18.1f}")
        print("="*86)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
