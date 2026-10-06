"""
Script de Benchmark Unificado para NVIDIA Jetson Orin Nano.

Ejecuta ambos métodos (Algoritmo Difuso Propuesto y WaSR-T) sobre los mismos frames
del video 'assets/videos/tesis.mp4' en la Jetson Orin Nano:
- Mide el procesamiento completo hasta obtener las cajas delimitadoras (end-to-end).
- Excluye dibujo, anotación gráfica y escritura de archivos de video.
- Excluye frames iniciales de calentamiento (warm-up).
- Si se usa GPU, asegura sincronización con torch.cuda.synchronize() antes de detener el cronómetro.
- Reporta tiempo medio por frame (ms) y throughput (FPS), especificando resolución, backend y modo de potencia.
- Guarda el registro completo en formato JSON y texto plano.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "WaSR-T"))

from src.pipeline_fuzzy_detector import detect_obstacles
from src.utils.utils import mask_to_bounding_boxes
import torch
from PIL import Image
from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights


def get_power_mode():
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "NV Power Mode" in line or "NVPM" in line:
                return line.strip()
        return out.strip().splitlines()[0]
    except Exception:
        return "Unknown (nvpmodel not available)"


def benchmark_fuzzy(video_path: str, num_frames: int = 60, warmup_frames: int = 10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video en {video_path}")

    times = []
    box_counts = []
    
    print(f"\n[+] Iniciando benchmark Difuso (Ours)...")
    print(f"    Total frames a leer: {num_frames} (Calentamiento: {warmup_frames})")

    for f_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        # Procesamiento completo hasta obtener cajas
        res = detect_obstacles(frame, target_size=(256, 192))
        boxes = res["confirmed_boxes"]
        t_elapsed = time.perf_counter() - t0

        if f_idx >= warmup_frames:
            times.append(t_elapsed)
            box_counts.append(len(boxes))

        if (f_idx + 1) % 20 == 0:
            print(f"    Frame {f_idx + 1}/{num_frames} procesado...")

    cap.release()

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    return {
        "method": "Ours (Fuzzy Logic)",
        "backend": "CPU (ARM Cortex-A78AE)",
        "resolution": "256x192",
        "total_frames_evaluated": len(times),
        "warmup_frames_excluded": warmup_frames,
        "mean_time_ms": mean_time * 1000.0,
        "std_time_ms": std_time * 1000.0,
        "fps": fps,
        "raw_times_sec": times
    }


def benchmark_fuzzy_gpu(video_path: str, num_frames: int = 60, warmup_frames: int = 10):
    from src.pipeline_fuzzy_detector_gpu import detect_obstacles_gpu
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video en {video_path}")

    times = []
    box_counts = []
    
    print(f"\n[+] Iniciando benchmark Difuso en GPU (PyTorch CUDA)...")
    print(f"    Total frames a leer: {num_frames} (Calentamiento: {warmup_frames})")

    for f_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        res = detect_obstacles_gpu(frame, target_size=(256, 192), device="cuda")
        torch.cuda.synchronize()
        boxes = res["confirmed_boxes"]
        t_elapsed = time.perf_counter() - t0

        if f_idx >= warmup_frames:
            times.append(t_elapsed)
            box_counts.append(len(boxes))

        if (f_idx + 1) % 20 == 0:
            print(f"    Frame {f_idx + 1}/{num_frames} procesado...")

    cap.release()

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    return {
        "method": "Ours (Fuzzy Logic)",
        "backend": "GPU (PyTorch CUDA)",
        "resolution": "256x192",
        "total_frames_evaluated": len(times),
        "warmup_frames_excluded": warmup_frames,
        "mean_time_ms": mean_time * 1000.0,
        "std_time_ms": std_time * 1000.0,
        "fps": fps,
        "raw_times_sec": times
    }


def benchmark_wasrt(video_path: str, weights_path: str, device_name: str = "cpu", num_frames: int = 60, warmup_frames: int = 10):
    device = torch.device(device_name)
    target_size = (512, 384)

    print(f"\n[+] Iniciando benchmark WaSR-T en {device_name.upper()}...")
    print(f"    Cargando modelo...")
    model = wasr_temporal_resnet101(pretrained=False, hist_len=5)
    state_dict = load_weights(weights_path)
    model.load_state_dict(state_dict)
    model = model.sequential().eval().to(device)
    model.clear_state()

    transform = PytorchHubNormalization()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video en {video_path}")

    times = []
    box_counts = []

    print(f"    Total frames a leer: {num_frames} (Calentamiento: {warmup_frames})")

    for f_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesamiento + inferencia + extracción de cajas
        t0 = time.perf_counter()

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_resized = img_pil.resize(target_size, Image.BILINEAR)
        img_normalized = transform(img_resized)
        input_batch = {'image': img_normalized.unsqueeze(0).to(device)}

        with torch.no_grad():
            res = model(input_batch)

        if device.type == "cuda":
            torch.cuda.synchronize()

        probs = res['out'].cpu().detach().numpy()
        pred_mask_idx = probs.argmax(1)[0].astype(np.uint8)
        obs_mask = (pred_mask_idx == 0).astype(np.uint8)
        boxes_wasrt = mask_to_bounding_boxes(obs_mask, min_area=30)

        t_elapsed = time.perf_counter() - t0

        if f_idx >= warmup_frames:
            times.append(t_elapsed)
            box_counts.append(len(boxes_wasrt))

        if (f_idx + 1) % 20 == 0:
            print(f"    Frame {f_idx + 1}/{num_frames} procesado...")

    cap.release()

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    backend_label = f"{'GPU (NVIDIA Ampere CUDA)' if device.type == 'cuda' else 'CPU (ARM Cortex-A78AE)'}"
    return {
        "method": "WaSR-T",
        "backend": backend_label,
        "resolution": f"{target_size[0]}x{target_size[1]}",
        "total_frames_evaluated": len(times),
        "warmup_frames_excluded": warmup_frames,
        "mean_time_ms": mean_time * 1000.0,
        "std_time_ms": std_time * 1000.0,
        "fps": fps,
        "raw_times_sec": times
    }


def main():
    video_file = project_root / "assets/videos/tesis.mp4"
    weights_file = project_root / "WaSR-T/wasrt_mastr1478.pth"

    power_mode = get_power_mode()
    print("=" * 70)
    print("  BENCHMARK DE VELOCIDAD EN NVIDIA JETSON ORIN NANO")
    print(f"  Modo de Potencia: {power_mode}")
    print(f"  Video de Entrada: {video_file.name}")
    print("=" * 70)

    # 1. Medición Difuso (Ours) en CPU
    res_fuzzy = benchmark_fuzzy(str(video_file), num_frames=60, warmup_frames=10)

    # 2. Medición Difuso (Ours) en GPU si CUDA está disponible
    if torch.cuda.is_available():
        res_fuzzy_gpu = benchmark_fuzzy_gpu(str(video_file), num_frames=60, warmup_frames=10)
    else:
        res_fuzzy_gpu = None

    # 3. Medición WaSR-T en CPU (para comparativa CPU-a-CPU equitativa)
    res_wasrt_cpu = benchmark_wasrt(str(video_file), str(weights_file), device_name="cpu", num_frames=10, warmup_frames=2)

    # 4. Medición WaSR-T en GPU si CUDA está disponible
    if torch.cuda.is_available():
        res_wasrt_gpu = benchmark_wasrt(str(video_file), str(weights_file), device_name="cuda", num_frames=60, warmup_frames=10)
    else:
        res_wasrt_gpu = None

    # Compilación de resultados
    all_results = {
        "platform": "NVIDIA Jetson Orin Nano Developer Kit (8GB)",
        "power_mode": power_mode,
        "video": str(video_file.name),
        "results": [res_fuzzy]
    }
    if res_fuzzy_gpu:
        all_results["results"].append(res_fuzzy_gpu)
    all_results["results"].append(res_wasrt_cpu)
    if res_wasrt_gpu:
        all_results["results"].append(res_wasrt_gpu)

    output_dir = project_root / "main_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_jetson_results.json"
    txt_path = output_dir / "benchmark_jetson_output.txt"

    # Guardar JSON sin los arrays gigantes de raw_times
    json_clean = {
        "platform": all_results["platform"],
        "power_mode": all_results["power_mode"],
        "video": all_results["video"],
        "benchmarks": [
            {k: v for k, v in r.items() if k != "raw_times_sec"}
            for r in all_results["results"]
        ]
    }
    with open(json_path, "w") as f:
        json.dump(json_clean, f, indent=2)

    # Generar reporte en texto plano
    report_lines = [
        "=" * 75,
        f"BENCHMARK DE RENDIMIENTO EN PLATAFORMA ÚNICA: {all_results['platform']}",
        f"Modo de Potencia: {power_mode}",
        f"Archivo de Video: {video_file.name}",
        "=" * 75,
        f"{'Método':<22} | {'Backend':<26} | {'Resolución':<10} | {'Tiempo Medio':<14} | {'Throughput (FPS)':<16}",
        "-" * 95
    ]
    for r in all_results["results"]:
        report_lines.append(
            f"{r['method']:<22} | {r['backend']:<26} | {r['resolution']:<10} | "
            f"{r['mean_time_ms']:>8.2f} ± {r['std_time_ms']:<4.1f} ms | {r['fps']:>14.2f} FPS"
        )
    report_lines.append("=" * 95)
    report_text = "\n".join(report_lines)

    with open(txt_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n[+] Resultados guardados en:")
    print(f"    - {json_path}")
    print(f"    - {txt_path}")


if __name__ == "__main__":
    main()
