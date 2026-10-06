"""
Generador del archivo de etiquetas unificado y tabla final de evaluación.
Revisa las detecciones cuadro a cuadro de ambos métodos sobre el video de tesis:
- Método difuso corregido (sin memoria temporal)
- WaSR-T (procesamiento secuencial con contexto temporal)

Garantiza:
1. Una única etiqueta real por frame (Ground Truth unificado).
2. Mismo número de frames evaluados para ambos métodos.
3. TP + FN idéntico para ambos.
4. FP + TN idéntico para ambos.
"""

import os
import sys
import json
import csv
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline_fuzzy_detector import detect_obstacles

def generate_labels():
    video_path = project_root / "assets/videos/tesis.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video en {video_path}")

    # Cargar datos previos de validación para extraer detecciones secuenciales de WaSR-T
    old_ann_file = project_root / "main_output/video_validation_results.json"
    with open(old_ann_file, "r") as f:
        old_data = json.load(f)
    old_anns = old_data.get("frame_annotations", {})

    step = 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_eval = [f for f in range(0, total_frames, step)]
    if 1799 not in frames_to_eval:
        frames_to_eval.append(1799)

    records = []
    eval_dict = {}

    print(f"[+] Procesando video con el pipeline difuso corregido sobre {len(frames_to_eval)} frames...")

    for i, f_idx in enumerate(frames_to_eval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Ground Truth unificado (presencia física en agua navegable)
        # Seg 1 (0-900): obstáculo presente
        # Seg 2 (910-1150): agua limpia / giro a mar abierto
        # Seg 3 (1160-1760): obstáculo presente
        # Seg 4 (1770-1799): agua limpia
        gt = bool((0 <= f_idx <= 900) or (1160 <= f_idx <= 1760))

        # 2. Detección difusa corregida (sin memoria temporal)
        det_fuzzy = detect_obstacles(frame)
        ours_det = bool(det_fuzzy["has_obstacle"])
        num_boxes_ours = int(len(det_fuzzy["confirmed_boxes"]))

        # 3. Detección de WaSR-T secuencial
        ann = old_anns.get(str(f_idx), {})
        wasrt_tag = ann.get("wasrt", "tn")
        wasrt_det = bool(wasrt_tag in ["tp", "fp"])

        # Clasificación binaria
        ours_class = ("TP" if ours_det else "FN") if gt else ("FP" if ours_det else "TN")
        wasrt_class = ("TP" if wasrt_det else "FN") if gt else ("FP" if wasrt_det else "TN")

        rec = {
            "frame_idx": int(f_idx),
            "gt_has_obstacle": bool(gt),
            "ours_detected": bool(ours_det),
            "ours_num_boxes": int(num_boxes_ours),
            "ours_classification": str(ours_class),
            "wasrt_detected": bool(wasrt_det),
            "wasrt_classification": str(wasrt_class)
        }
        records.append(rec)
        eval_dict[str(f_idx)] = rec

        if (i + 1) % 40 == 0:
            print(f"    Procesados {i + 1}/{len(frames_to_eval)} frames...")

    cap.release()

    # Cálculo riguroso de métricas
    def calc(classes):
        tp = sum(1 for c in classes if c == "TP")
        fp = sum(1 for c in classes if c == "FP")
        fn = sum(1 for c in classes if c == "FN")
        tn = sum(1 for c in classes if c == "TN")
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1_score": f1}

    stats_ours = calc([r["ours_classification"] for r in records])
    stats_wasrt = calc([r["wasrt_classification"] for r in records])

    out_dir = project_root / "main_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "etiquetas_evaluacion_video.json"
    csv_path = out_dir / "etiquetas_evaluacion_video.csv"
    tex_path = out_dir / "tabla_rendimiento.tex"
    md_path = out_dir / "tabla_rendimiento.md"

    # Guardar JSON
    output_json = {
        "metadata": {
            "description": "Evaluación unificada de presencia por frame en video tesis.mp4",
            "video_file": "assets/videos/tesis.mp4",
            "total_frames_evaluated": len(records),
            "roi_definition": "Superficie de agua navegable debajo del horizonte (y >= ajuste)",
            "gt_positive_frames": sum(1 for r in records if r["gt_has_obstacle"]),
            "gt_negative_frames": sum(1 for r in records if not r["gt_has_obstacle"]),
        },
        "metrics_ours": stats_ours,
        "metrics_wasrt": stats_wasrt,
        "frames": eval_dict
    }

    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # Guardar CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Generar tabla LaTeX
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Comparación Cuantitativa de Detección de Obstáculos en Video Real (Evaluación de Presencia por Frame)}}
\\label{{tab:video_performance_comparison}}
\\begin{{tabular}}{{lcccccccc}}
\\toprule
Método & Frames & TP & FP & FN & TN & Precisión & Recall & F1-Score \\\\
\\midrule
WaSR-T & {len(records)} & {stats_wasrt['tp']} & {stats_wasrt['fp']} & {stats_wasrt['fn']} & {stats_wasrt['tn']} & {stats_wasrt['precision']:.3f} & \\textbf{{{stats_wasrt['recall']:.3f}}} & \\textbf{{{stats_wasrt['f1_score']:.3f}}} \\\\
\\textbf{{Ours (Difuso)}} & {len(records)} & {stats_ours['tp']} & \\textbf{{{stats_ours['fp']}}} & {stats_ours['fn']} & \\textbf{{{stats_ours['tn']}}} & {stats_ours['precision']:.3f} & {stats_ours['recall']:.3f} & {stats_ours['f1_score']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(tex_path, "w") as f:
        f.write(latex_table)

    # Generar tabla Markdown
    md_table = f"""# Tabla Comparativa de Rendimiento (Presencia por Frame)

| Método | Frames | TP | FP | FN | TN | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Secuencial)** | {len(records)} | {stats_wasrt['tp']} | {stats_wasrt['fp']} | {stats_wasrt['fn']} | {stats_wasrt['tn']} | **{stats_wasrt['precision']:.3f}** | **{stats_wasrt['recall']:.3f}** | **{stats_wasrt['f1_score']:.3f}** |
| **Ours (Difuso sin memoria)** | {len(records)} | {stats_ours['tp']} | **{stats_ours['fp']}** | {stats_ours['fn']} | **{stats_ours['tn']}** | **{stats_ours['precision']:.3f}** | {stats_ours['recall']:.3f} | {stats_ours['f1_score']:.3f} |

### Verificaciones de Consistencia Metodológica:
- **Total de frames evaluados:** {len(records)} en ambos métodos.
- **TP + FN (Total Positivos GT):** Ours = {stats_ours['tp'] + stats_ours['fn']} | WaSR-T = {stats_wasrt['tp'] + stats_wasrt['fn']} (Coincidencia exacta).
- **FP + TN (Total Negativos GT):** Ours = {stats_ours['fp'] + stats_ours['tn']} | WaSR-T = {stats_wasrt['fp'] + stats_wasrt['tn']} (Coincidencia exacta).
"""
    with open(md_path, "w") as f:
        f.write(md_table)

    print("\n" + md_table)
    print(f"\n[+] Archivos generados:")
    print(f"    - {json_path}")
    print(f"    - {csv_path}")
    print(f"    - {tex_path}")
    print(f"    - {md_path}")

if __name__ == "__main__":
    generate_labels()
