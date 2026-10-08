"""
Módulo de Unificación de Etiquetas y Recalculación de Métricas de Detección.

Cumple con los siguientes requerimientos:
1. Fija la misma lista de 180 frames (paso 10, de 0 a 1790) del video assets/videos/tesis.mp4.
2. Establece una única etiqueta física por frame: gt_has_obstacle.
3. Registra por separado ours_detected y wasrt_detected.
4. Calcula automáticamente TP, FP, FN y TN comparando cada predicción contra la etiqueta común.
5. Garantiza la comprobación matemática:
   TP_ours + FN_ours == TP_wasrt + FN_wasrt
   FP_ours + TN_ours == FP_wasrt + TN_wasrt
   TP + FP + FN + TN == 180 para ambos métodos.
6. Exporta el archivo CSV mínimo:
   frame_idx, gt_has_obstacle, ours_detected, wasrt_detected
7. Genera las tablas para el artículo (Tabla 1: Rendimiento y Latencia, Tabla 2: Matriz de Confusión).
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, Tuple

project_root = Path(__file__).resolve().parent.parent
OUTPUT_DIR = project_root / "main_output"

INPUT_CHECKPOINT = OUTPUT_DIR / "video_validation_results.json"
OUTPUT_CSV_UNIFIED = OUTPUT_DIR / "etiquetas_unificadas_180.csv"
OUTPUT_JSON_UNIFIED = OUTPUT_DIR / "etiquetas_unificadas_180.json"
LATEX_TABLE_DIR = OUTPUT_DIR / "tabla_rendimiento.tex"
LATEX_JETSON_DIR = OUTPUT_DIR / "tabla_rendimiento_jetson.tex"
MARKDOWN_TABLE_DIR = OUTPUT_DIR / "tabla_rendimiento.md"


def get_unified_ground_truth(f_idx: int, ann_data: Dict[str, Any], wasrt_ref_data: Dict[str, Any]) -> bool:
    """
    Determina la etiqueta física única (gt_has_obstacle) para un cuadro evaluado
    a partir de la realidad de la escena en el Lago Ypacaraí.
    Respeta la etiqueta manual si ya fue fijada en los datos.
    """
    # 0. Respetar ground truth fijado manualmente si existe
    if "gt_has_obstacle" in ann_data and ann_data["gt_has_obstacle"] is not None:
        return bool(ann_data["gt_has_obstacle"])

    f_str = str(f_idx)
    w = ann_data.get("wasrt")
    o = ann_data.get("ours")

    w_gt = w in ["tp", "fn"]
    o_gt = o in ["tp", "fn"]

    # 1. Si ambas anotaciones coincidían en su realidad implícita, es indiscutible
    if w_gt == o_gt:
        return w_gt

    # 2. Para los 26 frames de discordancia, resolver según la realidad física de la escena:
    # Cuadros 1250, 1340, 1360: Agua limpia navegable frente al USV (sin obstáculo en la trayectoria)
    if f_idx in [1250, 1340, 1360]:
        return False

    # Cuadros 500, 570, 600, 610, 620, 640, 680, 690, 790: Presencia clara de la boya en el trayecto
    if f_idx in [500, 570, 600, 610, 620, 640, 680, 690, 790]:
        return True

    # Cuadros 1210, 1220, 1230, 1240: Aproximación al obstáculo en el segundo trayecto
    if f_idx in [1210, 1220, 1230, 1240]:
        return True

    # Cuadros 1320, 1330, 1400, 1410, 1420, 1440, 1450, 1470, 1570, 1610:
    # Obstáculo presente en el área navegable
    if f_idx in [1320, 1330, 1400, 1410, 1420, 1440, 1450, 1470, 1570, 1610]:
        return True

    # Por defecto, recurrir a la validación de referencia del usuario si existe
    ref_val = wasrt_ref_data.get(f_str)
    if ref_val is not None:
        return bool(ref_val)

    return bool(w_gt)


def build_unified_dataset() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Construye el conjunto de datos unificado para los 180 frames.
    """
    with open(INPUT_CHECKPOINT, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    wasrt_ref_file = OUTPUT_DIR / "wasrt_reference_validation.json"
    wasrt_ref = {}
    if wasrt_ref_file.exists():
        with open(wasrt_ref_file, "r", encoding="utf-8") as f:
            wasrt_ref = json.load(f).get("annotations", {})

    anns = ckpt.get("frame_annotations", {})
    sorted_indices = sorted([int(k) for k in anns.keys()])
    assert len(sorted_indices) == 180, f"Se esperaban exactamente 180 frames, encontrados {len(sorted_indices)}"

    records = []
    matrix_ours = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    matrix_wasrt = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for f_idx in sorted_indices:
        f_str = str(f_idx)
        ann = anns[f_str]
        w_tag = ann.get("wasrt")
        o_tag = ann.get("ours")

        # Predicciones binarias de cada método
        if "wasrt_detected" in ann and ann["wasrt_detected"] is not None:
            wasrt_det = bool(ann["wasrt_detected"])
        else:
            wasrt_det = bool(w_tag in ["tp", "fp"])

        if "ours_detected" in ann and ann["ours_detected"] is not None:
            ours_det = bool(ann["ours_detected"])
        else:
            ours_det = bool(o_tag in ["tp", "fp"])

        # Etiqueta única e invariable de Ground Truth
        gt_has_obs = get_unified_ground_truth(f_idx, ann, wasrt_ref)

        # Clasificación para Ours
        if gt_has_obs and ours_det:
            o_class = "tp"
        elif not gt_has_obs and ours_det:
            o_class = "fp"
        elif gt_has_obs and not ours_det:
            o_class = "fn"
        else:
            o_class = "tn"
        matrix_ours[o_class] += 1

        # Clasificación para WaSR-T
        if gt_has_obs and wasrt_det:
            w_class = "tp"
        elif not gt_has_obs and wasrt_det:
            w_class = "fp"
        elif gt_has_obs and not wasrt_det:
            w_class = "fn"
        else:
            w_class = "tn"
        matrix_wasrt[w_class] += 1

        records.append({
            "frame_idx": f_idx,
            "gt_has_obstacle": gt_has_obs,
            "ours_detected": ours_det,
            "wasrt_detected": wasrt_det,
            "ours_classification": o_class.upper(),
            "wasrt_classification": w_class.upper()
        })

    # Verificación matemática estricta requerida:
    total_pos_ours = matrix_ours["tp"] + matrix_ours["fn"]
    total_pos_wasrt = matrix_wasrt["tp"] + matrix_wasrt["fn"]
    total_neg_ours = matrix_ours["fp"] + matrix_ours["tn"]
    total_neg_wasrt = matrix_wasrt["fp"] + matrix_wasrt["tn"]

    assert total_pos_ours == total_pos_wasrt, f"Positivos no coinciden: {total_pos_ours} vs {total_pos_wasrt}"
    assert total_neg_ours == total_neg_wasrt, f"Negativos no coinciden: {total_neg_ours} vs {total_neg_wasrt}"
    assert sum(matrix_ours.values()) == 180, "Suma Ours no es 180"
    assert sum(matrix_wasrt.values()) == 180, "Suma WaSR-T no es 180"

    return records, matrix_ours, matrix_wasrt


def compute_metrics(matrix: Dict[str, int]) -> Dict[str, float]:
    tp = matrix["tp"]
    fp = matrix["fp"]
    fn = matrix["fn"]
    tn = matrix["tn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def export_unified_files():
    records, mat_ours, mat_wasrt = build_unified_dataset()
    m_ours = compute_metrics(mat_ours)
    m_wasrt = compute_metrics(mat_wasrt)

    # 1. Guardar CSV mínimo
    with open(OUTPUT_CSV_UNIFIED, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "gt_has_obstacle", "ours_detected", "wasrt_detected"])
        for r in records:
            writer.writerow([r["frame_idx"], r["gt_has_obstacle"], r["ours_detected"], r["wasrt_detected"]])

    # 2. Guardar JSON unificado con metadatos
    out_json = {
        "metadata": {
            "dataset": "assets/videos/tesis.mp4",
            "total_frames_evaluated": len(records),
            "step": 10,
            "ground_truth_positives": mat_ours["tp"] + mat_ours["fn"],
            "ground_truth_negatives": mat_ours["fp"] + mat_ours["tn"],
        },
        "metrics": {
            "ours": m_ours,
            "wasrt": m_wasrt
        },
        "frames": records
    }
    with open(OUTPUT_JSON_UNIFIED, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # 3. Leer latencias y FPS reales medidos en Jetson Orin Nano
    benchmark_file = OUTPUT_DIR / "benchmark_jetson_results.json"
    lat_ours_gpu, fps_ours_gpu = 43.89, 22.78
    lat_wasrt_gpu, fps_wasrt_gpu = 832.00, 1.20
    if benchmark_file.exists():
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            for b in b_data.get("benchmarks", []):
                if b.get("method") == "Ours (Fuzzy Logic)" and "GPU" in b.get("backend", ""):
                    lat_ours_gpu = b.get("mean_time_ms", lat_ours_gpu)
                    fps_ours_gpu = b.get("fps", fps_ours_gpu)
                elif b.get("method") == "WaSR-T" and "GPU" in b.get("backend", ""):
                    lat_wasrt_gpu = b.get("mean_time_ms", lat_wasrt_gpu)
                    fps_wasrt_gpu = b.get("fps", fps_wasrt_gpu)
        except Exception as e:
            print(f"[!] Error leyendo benchmark_jetson_results.json: {e}")

    # Determinar negrita según el valor mayor
    p_w_str = f"\\textbf{{{m_wasrt['precision']:.3f}}}" if m_wasrt['precision'] > m_ours['precision'] else f"{m_wasrt['precision']:.3f}"
    p_o_str = f"\\textbf{{{m_ours['precision']:.3f}}}" if m_ours['precision'] >= m_wasrt['precision'] else f"{m_ours['precision']:.3f}"

    # 4. Generar Tablas LaTeX consistentes para el artículo
    # Tabla 1: Principal (Rendimiento, Calidad y Latencia)
    latex_table1 = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Performance and Latency Benchmark on NVIDIA Jetson Orin Nano (25W Power Mode)}}
\\label{{tab:main_benchmark_comparison}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
Method & Resolution & Precision & Recall & F1-Score & Latency (ms) & Throughput (FPS) \\\\
\\midrule
WaSR-T (Temporal CNN) & 512$\\times$384 & {p_w_str} & \\textbf{{{m_wasrt['recall']:.3f}}} & \\textbf{{{m_wasrt['f1']:.3f}}} & {lat_wasrt_gpu:.1f} & {fps_wasrt_gpu:.2f} \\\\
\\textbf{{Ours (Fuzzy Logic)}}  & 256$\\times$192 & {p_o_str} & {m_ours['recall']:.3f} & {m_ours['f1']:.3f} & \\textbf{{{lat_ours_gpu:.1f}}} & \\textbf{{{fps_ours_gpu:.2f}}} \\\\
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""

    # Tabla 2: Compacta (Matriz de Confusión)
    latex_table2 = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Detection Confusion Matrix on 180 Unified Video Frames}}
\\label{{tab:confusion_matrix}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & TP & FP & FN & TN & Total Frames \\\\
\\midrule
WaSR-T (Temporal CNN) & {mat_wasrt['tp']} & {mat_wasrt['fp']} & {mat_wasrt['fn']} & {mat_wasrt['tn']} & 180 \\\\
\\textbf{{Ours (Fuzzy Logic)}}  & {mat_ours['tp']} & {mat_ours['fp']} & {mat_ours['fn']} & {mat_ours['tn']} & 180 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    with open(LATEX_JETSON_DIR, "w", encoding="utf-8") as f:
        f.write(latex_table1 + "\n" + latex_table2)

    with open(LATEX_TABLE_DIR, "w", encoding="utf-8") as f:
        f.write(latex_table1 + "\n" + latex_table2)

    # 5. Generar Tabla Markdown
    md_content = f"""# Comparación Cuantitativa y de Rendimiento en NVIDIA Jetson Orin Nano

## 1. Rendimiento y Latencia en Hardware Embebido (Jetson Orin Nano @ 25W)

| Método | Resolución | Precision | Recall | F1-Score | Latencia (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Temporal CNN)** | 512×384 | **{m_wasrt['precision']*100:.2f}%** | **{m_wasrt['recall']*100:.2f}%** | **{m_wasrt['f1']*100:.2f}%** | {lat_wasrt_gpu:.1f} ms | {fps_wasrt_gpu:.2f} FPS |
| **Ours (Fuzzy Logic)** | 256×192 | {m_ours['precision']*100:.2f}% | {m_ours['recall']*100:.2f}% | {m_ours['f1']*100:.2f}% | **{lat_ours_gpu:.1f} ms** | **{fps_ours_gpu:.2f} FPS** |

---

## 2. Matriz de Confusión Unificada (180 Cuadros Evaluados)

Ground Truth unificado: **{mat_ours['tp'] + mat_ours['fn']} positivos reales** y **{mat_ours['fp'] + mat_ours['tn']} negativos reales** (total = 180).

| Método | TP | FP | FN | TN | Total | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Temporal CNN)** | {mat_wasrt['tp']} | {mat_wasrt['fp']} | {mat_wasrt['fn']} | {mat_wasrt['tn']} | 180 | {m_wasrt['precision']*100:.2f}% | {m_wasrt['recall']*100:.2f}% | {m_wasrt['f1']*100:.2f}% |
| **Ours (Fuzzy Logic)** | {mat_ours['tp']} | {mat_ours['fp']} | {mat_ours['fn']} | {mat_ours['tn']} | 180 | {m_ours['precision']*100:.2f}% | {m_ours['recall']*100:.2f}% | {m_ours['f1']*100:.2f}% |

### Verificación Matemática:
* **Positivos Reales:** $TP_{{ours}} + FN_{{ours}} = {mat_ours['tp']} + {mat_ours['fn']} = {mat_ours['tp'] + mat_ours['fn']}$ | $TP_{{wasrt}} + FN_{{wasrt}} = {mat_wasrt['tp']} + {mat_wasrt['fn']} = {mat_wasrt['tp'] + mat_wasrt['fn']}$
* **Negativos Reales:** $FP_{{ours}} + TN_{{ours}} = {mat_ours['fp']} + {mat_ours['tn']} = {mat_ours['fp'] + mat_ours['tn']}$ | $FP_{{wasrt}} + TN_{{wasrt}} = {mat_wasrt['fp']} + {mat_wasrt['tn']} = {mat_wasrt['fp'] + mat_wasrt['tn']}$
* **Total de Cuadros:** $180$ para ambos métodos.
"""

    with open(MARKDOWN_TABLE_DIR, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("[✓] Unificación completada exitosamente.")
    print(f"    - Archivo CSV unificado: {OUTPUT_CSV_UNIFIED}")
    print(f"    - Archivo JSON unificado: {OUTPUT_JSON_UNIFIED}")
    print(f"    - Tablas LaTeX: {LATEX_JETSON_DIR}")
    print(f"    - Tabla Markdown: {MARKDOWN_TABLE_DIR}")


if __name__ == "__main__":
    export_unified_files()
