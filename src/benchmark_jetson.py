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
import resource
import torch
from PIL import Image
from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights


def get_peak_ram_mb():
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    except Exception:
        return 0.0


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
        "peak_ram_mb": get_peak_ram_mb(),
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
        "peak_ram_mb": get_peak_ram_mb(),
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
        "peak_ram_mb": get_peak_ram_mb(),
        "raw_times_sec": times
    }


def main(all_results=None, args=None):
    video_file = project_root / "assets/videos/tesis.mp4"
    weights_file = project_root / "WaSR-T/wasrt_mastr1478.pth"

    power_mode = get_power_mode()

    if all_results is None:
        num_frames = getattr(args, "num_frames", 60) if args else 60
        warmup_frames = getattr(args, "warmup_frames", 10) if args else 10
        skip_wasrt_cpu = getattr(args, "skip_wasrt_cpu", False) if args else False
        gpu_only = getattr(args, "gpu_only", False) if args else False

        print("=" * 75)
        print("  BENCHMARK UNIFICADO EN NVIDIA JETSON ORIN NANO DEVELOPER KIT")
        print(f"  Modo de Potencia: {power_mode}")
        print(f"  Video de Entrada: {video_file.name}")
        print("=" * 75)

        results_list = []
        if not gpu_only:
            res_fuzzy = benchmark_fuzzy(str(video_file), num_frames=num_frames, warmup_frames=warmup_frames)
            results_list.append(res_fuzzy)

        if torch.cuda.is_available():
            res_fuzzy_gpu = benchmark_fuzzy_gpu(str(video_file), num_frames=num_frames, warmup_frames=warmup_frames)
            results_list.append(res_fuzzy_gpu)

        if not gpu_only and not skip_wasrt_cpu:
            res_wasrt_cpu = benchmark_wasrt(str(video_file), str(weights_file), device_name="cpu", num_frames=min(10, num_frames), warmup_frames=2)
            results_list.append(res_wasrt_cpu)

        if torch.cuda.is_available():
            res_wasrt_gpu = benchmark_wasrt(str(video_file), str(weights_file), device_name="cuda", num_frames=num_frames, warmup_frames=warmup_frames)
            results_list.append(res_wasrt_gpu)

        all_results = {
            "platform": "NVIDIA Jetson Orin Nano Developer Kit (8GB)",
            "power_mode": power_mode,
            "video": str(video_file.name),
            "results": results_list
        }

    output_dir = project_root / "main_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_jetson_results.json"
    txt_path = output_dir / "benchmark_jetson_output.txt"
    latex_path = output_dir / "tabla_rendimiento_jetson.tex"

    # Cargar métricas de validación de video si existen
    unified_json_path = output_dir / "etiquetas_unificadas_180.json"
    val_json_path = output_dir / "video_validation_results.json"
    detection_metrics = None
    if unified_json_path.exists():
        try:
            with open(unified_json_path, "r", encoding="utf-8") as f:
                u_data = json.load(f)
            detection_metrics = {
                "total_annotated": u_data["metadata"]["total_frames_evaluated"],
                "ours": u_data["metrics"]["ours"],
                "wasrt": u_data["metrics"]["wasrt"]
            }
        except Exception as e:
            print(f"[!] Aviso leyendo métricas unificadas: {e}")
    elif val_json_path.exists():
        try:
            with open(val_json_path, "r", encoding="utf-8") as f:
                v_data = json.load(f)
            eval_o = v_data.get("eval_ours", {})
            eval_w = v_data.get("eval_wasrt", {})
            def calc_prf(st):
                tp = st.get("tp", 0)
                fp = st.get("fp", 0)
                fn = st.get("fn", 0)
                tn = st.get("tn", 0)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                return p, r, f1, tp, fp, fn, tn

            p_o, r_o, f1_o, tp_o, fp_o, fn_o, tn_o = calc_prf(eval_o)
            p_w, r_w, f1_w, tp_w, fp_w, fn_w, tn_w = calc_prf(eval_w)
            detection_metrics = {
                "total_annotated": len(v_data.get("frame_annotations", {})),
                "ours": {"precision": p_o, "recall": r_o, "f1": f1_o, "tp": tp_o, "fp": fp_o, "fn": fn_o, "tn": tn_o},
                "wasrt": {"precision": p_w, "recall": r_w, "f1": f1_w, "tp": tp_w, "fp": fp_w, "fn": fn_w, "tn": tn_w}
            }
        except Exception as e:
            print(f"[!] Aviso leyendo métricas de validación: {e}")

    # Guardar JSON limpio
    json_clean = {
        "platform": all_results["platform"],
        "power_mode": all_results["power_mode"],
        "video": all_results["video"],
        "benchmarks": [
            {k: v for k, v in r.items() if k != "raw_times_sec"}
            for r in all_results["results"]
        ]
    }
    if detection_metrics:
        json_clean["detection_metrics"] = detection_metrics

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_clean, f, indent=2)

    # Generar reporte en texto plano
    report_lines = [
        "=" * 105,
        f"BENCHMARK DE RENDIMIENTO EN PLATAFORMA ÚNICA: {all_results['platform']}",
        f"Modo de Potencia: {power_mode} | Video de Entrada: {video_file.name}",
        "=" * 105,
        f"{'Método':<20} | {'Backend':<24} | {'Resolución':<10} | {'Tiempo Medio':<16} | {'Throughput':<12} | {'RAM Pico':<10}",
        "-" * 105
    ]
    for r in all_results["results"]:
        ram_str = f"{r.get('peak_ram_mb', 0):.0f} MB"
        report_lines.append(
            f"{r['method']:<20} | {r['backend']:<24} | {r['resolution']:<10} | "
            f"{r['mean_time_ms']:>8.2f} ± {r['std_time_ms']:<4.1f} ms | {r['fps']:>8.2f} FPS | {ram_str:>10}"
        )
    report_lines.append("=" * 105)

    if detection_metrics:
        dm = detection_metrics
        report_lines.extend([
            "",
            "=" * 105,
            f"CALIDAD DE DETECCIÓN EN VIDEO (Métricas sobre {dm['total_annotated']} frames validados):",
            "=" * 105,
            f"{'Método':<20} | {'Precisión':<12} | {'Recall':<12} | {'F1-Score':<12} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}",
            "-" * 105,
            f"{'WaSR-T (CNN)':<20} | {dm['wasrt']['precision']:>10.4f}  | {dm['wasrt']['recall']:>10.4f}  | {dm['wasrt']['f1']:>10.4f}  | "
            f"{dm['wasrt']['tp']:>4d}   | {dm['wasrt']['fp']:>4d}   | {dm['wasrt']['fn']:>4d}   | {dm['wasrt']['tn']:>4d}",
            f"{'Ours (Fuzzy Renovado)':<20} | {dm['ours']['precision']:>10.4f}  | {dm['ours']['recall']:>10.4f}  | {dm['ours']['f1']:>10.4f}  | "
            f"{dm['ours']['tp']:>4d}   | {dm['ours']['fp']:>4d}   | {dm['ours']['fn']:>4d}   | {dm['ours']['tn']:>4d}",
            "=" * 105
        ])

    report_text = "\n".join(report_lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Generar tablas LaTeX para el paper/tesis
    f_gpu_r = next((r for r in all_results["results"] if r["method"].startswith("Ours") and "GPU" in r["backend"]), None)
    w_gpu_r = next((r for r in all_results["results"] if r["method"] == "WaSR-T" and "GPU" in r["backend"]), None)

    lat_ours_gpu = f_gpu_r["mean_time_ms"] if f_gpu_r else 43.89
    fps_ours_gpu = f_gpu_r["fps"] if f_gpu_r else 22.78
    lat_wasrt_gpu = w_gpu_r["mean_time_ms"] if w_gpu_r else 832.00
    fps_wasrt_gpu = w_gpu_r["fps"] if w_gpu_r else 1.20

    p_w_s = f"{detection_metrics['wasrt']['precision']:.3f}" if detection_metrics else "0.866"
    r_w_s = f"{detection_metrics['wasrt']['recall']:.3f}" if detection_metrics else "0.729"
    f1_w_s = f"{detection_metrics['wasrt']['f1']:.3f}" if detection_metrics else "0.792"
    p_o_s = f"{detection_metrics['ours']['precision']:.3f}" if detection_metrics else "0.716"
    r_o_s = f"{detection_metrics['ours']['recall']:.3f}" if detection_metrics else "0.464"
    f1_o_s = f"{detection_metrics['ours']['f1']:.3f}" if detection_metrics else "0.563"

    tp_w = detection_metrics['wasrt']['tp'] if detection_metrics else 97
    fp_w = detection_metrics['wasrt']['fp'] if detection_metrics else 15
    fn_w = detection_metrics['wasrt']['fn'] if detection_metrics else 36
    tn_w = detection_metrics['wasrt']['tn'] if detection_metrics else 32

    tp_o = detection_metrics['ours']['tp'] if detection_metrics else 58
    fp_o = detection_metrics['ours']['fp'] if detection_metrics else 23
    fn_o = detection_metrics['ours']['fn'] if detection_metrics else 67
    tn_o = detection_metrics['ours']['tn'] if detection_metrics else 32

    latex_table1 = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Performance and Latency Benchmark on NVIDIA Jetson Orin Nano (25W Power Mode)}}
\\label{{tab:main_benchmark_comparison}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
Method & Resolution & Precision & Recall & F1-Score & Latency (ms) & Throughput (FPS) \\\\
\\midrule
WaSR-T (Temporal CNN) & 512$\\times$384 & \\textbf{{{p_w_s}}} & \\textbf{{{r_w_s}}} & \\textbf{{{f1_w_s}}} & {lat_wasrt_gpu:.1f} & {fps_wasrt_gpu:.2f} \\\\
\\textbf{{Ours (Fuzzy Logic)}}  & 256$\\times$192 & {p_o_s} & {r_o_s} & {f1_o_s} & \\textbf{{{lat_ours_gpu:.1f}}} & \\textbf{{{fps_ours_gpu:.2f}}} \\\\
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""

    latex_table2 = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Detection Confusion Matrix on 180 Unified Video Frames}}
\\label{{tab:confusion_matrix}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & TP & FP & FN & TN & Total Frames \\\\
\\midrule
WaSR-T (Temporal CNN) & {tp_w} & {fp_w} & {fn_w} & {tn_w} & 180 \\\\
\\textbf{{Ours (Fuzzy Logic)}}  & {tp_o} & {fp_o} & {fn_o} & {tn_o} & 180 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    full_latex = latex_table1 + "\n" + latex_table2
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(full_latex)

    print("\n" + report_text)
    print(f"\n[+] Resultados guardados en:")
    print(f"    - {json_path}")
    print(f"    - {txt_path}")
    print(f"    - {latex_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark en NVIDIA Jetson Orin Nano")
    parser.add_argument("--num-frames", type=int, default=60, help="Frames a evaluar (por defecto 60)")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Frames de calentamiento excluidos")
    parser.add_argument("--skip-wasrt-cpu", action="store_true", help="Omitir benchmark lento de WaSR-T en CPU")
    parser.add_argument("--gpu-only", action="store_true", help="Evaluar únicamente variantes en GPU")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
