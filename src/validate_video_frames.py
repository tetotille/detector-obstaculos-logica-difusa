import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
import numpy as np
import sys
import json
from pathlib import Path

# Configuración de rutas
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.utils.utils import mask_to_bounding_boxes
from src.pipeline_fuzzy_detector import detect_obstacles
from src.comparar_deteccion_imagen import process_ours

OUTPUT_DIR = project_root / "main_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "video_validation_results.json"
LATEX_TABLE_FILE = OUTPUT_DIR / "tabla_rendimiento.tex"

VIDEO_ORIGINAL = project_root / "assets/videos/tesis.mp4"
VIDEO_OURS = OUTPUT_DIR / "tesis_procesado.mp4"
VIDEO_WASRT = OUTPUT_DIR / "tesis_wasrt_cpu.mp4"


def recompute_annotations_headless(state):
    """
    Recalcula automáticamente las clasificaciones de 'ours' para todos los frames
    anotados en video_validation_results.json utilizando el algoritmo renovado.
    Determina el ground truth a partir de las validaciones de WaSR-T o etiquetas previas.
    """
    cap = cv2.VideoCapture(str(VIDEO_ORIGINAL))
    if not cap.isOpened():
        print(f"[!] No se pudo abrir {VIDEO_ORIGINAL}")
        return

    anns = state.get("frame_annotations", {})
    print(f"\n[+] Recalculando {len(anns)} frames anotados con el algoritmo renovado...")
    
    tp_o, fp_o, fn_o, tn_o = 0, 0, 0, 0
    count = 0
    for idx_str, ann in sorted(anns.items(), key=lambda x: int(x[0])):
        f_idx = int(idx_str)
        w = ann.get("wasrt")
        o = ann.get("ours")
        
        # Determinar ground truth objetivo del frame
        if w in ["tp", "fn"]:
            gt_has_obs = True
        elif w in ["tn", "fp"]:
            gt_has_obs = False
        elif o in ["tp", "fn"]:
            gt_has_obs = True
        elif o in ["tn", "fp"]:
            gt_has_obs = False
        else:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        res = detect_obstacles(frame, target_size=(256, 192))
        has_obs_ours = res["has_obstacle"]
        
        if gt_has_obs and has_obs_ours:
            tag = "tp"
            tp_o += 1
        elif not gt_has_obs and has_obs_ours:
            tag = "fp"
            fp_o += 1
        elif gt_has_obs and not has_obs_ours:
            tag = "fn"
            fn_o += 1
        else:
            tag = "tn"
            tn_o += 1
            
        ann["ours"] = tag
        count += 1
        if count % 25 == 0 or count == len(anns):
            print(f"  Progreso: {count}/{len(anns)} frames | TP={tp_o}, FP={fp_o}, FN={fn_o}, TN={tn_o}")

    cap.release()
    state["eval_ours"] = {"tp": tp_o, "fp": fp_o, "fn": fn_o, "tn": tn_o}
    save_checkpoint(state)
    
    p_o, r_o, f1_o, tot_o = calc_metrics(state["eval_ours"])
    p_w, r_w, f1_w, tot_w = calc_metrics(state["eval_wasrt"])
    
    print("\n" + "=" * 70)
    print("  RESULTADOS ACTUALIZADOS CON EL ALGORITMO RENOVADO")
    print("=" * 70)
    print(f"Frames evaluados: {tot_o}")
    print(f"Ours (Fuzzy Logic) : TP={tp_o:3d} | FP={fp_o:3d} | FN={fn_o:3d} | TN={tn_o:3d} | Prec={p_o:.4f} | Rec={r_o:.4f} | F1={f1_o:.4f}")
    print(f"WaSR-T (CNN)       : TP={state['eval_wasrt']['tp']:3d} | FP={state['eval_wasrt']['fp']:3d} | FN={state['eval_wasrt']['fn']:3d} | TN={state['eval_wasrt']['tn']:3d} | Prec={p_w:.4f} | Rec={r_w:.4f} | F1={f1_w:.4f}")
    print("=" * 70)
    
    tbl = generate_latex_table(state["eval_ours"], state["eval_wasrt"])
    print("\nTabla LaTeX generada en:", LATEX_TABLE_FILE)
    print(tbl)

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
                print(f"[+] Checkpoint cargado desde {CHECKPOINT_FILE}")
                return data
        except Exception as e:
            print(f"[!] Error leyendo checkpoint: {e}")
    
    return {
        "last_frame": 0,
        "step": 10,  # Salto de frames por defecto
        "eval_ours": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "eval_wasrt": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "frame_annotations": {}  # frame_idx -> {"gt_has_obs": bool, "ours": "tp/fp/fn/tn", "wasrt": "tp/fp/fn/tn"}
    }

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def calc_metrics(stats):
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    total = tp + fp + fn + tn
    return precision, recall, f1, total

def generate_latex_table(stats_ours, stats_wasrt, fps_ours=10.41, fps_wasrt=2.21, ram_ours=242, ram_wasrt=1685):
    p_o, r_o, f1_o, tot_o = calc_metrics(stats_ours)
    p_w, r_w, f1_w, tot_w = calc_metrics(stats_wasrt)
    
    # Formateo con negritas dinámicas para el mejor valor
    def b_max(val_o, val_w, fmt=".3f"):
        s_o = f"{val_o:{fmt}}"
        s_w = f"{val_w:{fmt}}"
        if val_o > val_w:
            return f"\\textbf{{{s_o}}}", s_w
        elif val_w > val_o:
            return s_o, f"\\textbf{{{s_w}}}"
        return s_o, s_w

    def b_min(val_o, val_w):
        s_o = f"{val_o}"
        s_w = f"{val_w}"
        if val_o < val_w:
            return f"\\textbf{{{s_o}}}", s_w
        elif val_w < val_o:
            return s_o, f"\\textbf{{{s_w}}}"
        return s_o, s_w

    prec_o, prec_w = b_max(p_o, p_w)
    rec_o, rec_w = b_max(r_o, r_w)
    f1_o_str, f1_w_str = b_max(f1_o, f1_w)
    fps_o, fps_w = b_max(fps_ours, fps_wasrt, fmt=".2f")
    ram_o, ram_w = b_min(ram_ours, ram_wasrt)

    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative Performance and Resource Consumption Comparison on CPU Execution}}
\\label{{tab:performance_comparison}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & Precision & Recall & F1-Score & Throughput (FPS) & Peak RAM (MB) \\\\
\\midrule
WaSR-T & {prec_w} & {rec_w} & {f1_w_str} & {fps_w} & {ram_w} \\\\
\\textbf{{Ours}}  & {prec_o} & {rec_o} & {f1_o_str} & {fps_o} & {ram_o} \\\\
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    with open(LATEX_TABLE_FILE, "w") as f:
        f.write(latex_code)
    
    return latex_code

def main():
    parser = argparse.ArgumentParser(description="Validación interactiva o desatendida de frames de video")
    parser.add_argument("--auto-eval", action="store_true", help="Recalcula las métricas de 'ours' de forma desatendida y genera la tabla LaTeX")
    args = parser.parse_args()

    state = load_checkpoint()
    if args.auto_eval:
        recompute_annotations_headless(state)
        return

    print("=" * 70)
    print("  HERRAMIENTA DE VALIDACIÓN INTERACTIVA DE VIDEO (FRAME POR FRAME)")
    print("=" * 70)
    
    # Check videos
    for v_name, v_path in [("Original", VIDEO_ORIGINAL), ("WaSR-T", VIDEO_WASRT)]:
        if not v_path.exists():
            print(f"[!] ERROR: Video {v_name} not found at: {v_path}")
            return
    
    cap_orig = cv2.VideoCapture(str(VIDEO_ORIGINAL))
    cap_wasrt = cv2.VideoCapture(str(VIDEO_WASRT))
    
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap_orig.get(cv2.CAP_PROP_FPS)
    
    current_frame = state.get("last_frame", 0)
    step = state.get("step", 10)
    show_masks = False
    
    window_name = "Comparative Validation: Original | Ours (Fuzzy Logic) | WaSR-T"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1500, 520)
    
    print("\n[INSTRUCCIONES DE TECLADO]")
    print("  -------------------------------------------------------------")
    print("  [ACCESOS RÁPIDOS COMBINADOS]:")
    print("    'b' : Ambos detectaron bien el obstáculo (Ours=TP, WaSR-T=TP)")
    print("    '0' : Ambos agua limpia correcta (Ours=TN, WaSR-T=TN)")
    print("  -------------------------------------------------------------")
    print("  [LÓGICA DIFUSA (OURS)]:")
    print("    '1' : TP (Obstáculo presente y bien detectado)")
    print("    '2' : FP (Falsa alarma en agua donde no hay obstáculo)")
    print("    '3' : FN (Había obstáculo y NO lo detectó)")
    print("    '4' : TN (Agua limpia y no detectó nada)")
    print("  -------------------------------------------------------------")
    print("  [WaSR-T]:")
    print("    'q' : TP (Obstáculo presente y bien detectado)")
    print("    'w' : FP (Falsa alarma en agua)")
    print("    'e' : FN (Había obstáculo y NO lo detectó)")
    print("    'r' : TN (Agua limpia)")
    print("  -------------------------------------------------------------")
    print("  [NAVEGACIÓN Y EDICIÓN]:")
    print("    'c'             : Limpiar / Borrar etiqueta del frame actual")
    print("    [ESPACIO] o 'd' : Siguiente frame")
    print("    'a'             : Frame anterior")
    print("    '+' / '-'       : Aumentar / Disminuir salto de frames (actual: {})".format(step))
    print("    's'             : Guardar checkpoint y mostrar tabla LaTeX")
    print("    [ESC] o 'x'     : Guardar y salir")
    print("  -------------------------------------------------------------\n")
    
    def fetch_frames(idx):
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_wasrt.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        ret1, f_orig = cap_orig.read()
        ret3, f_wasrt = cap_wasrt.read()
        return ret1 and ret3, f_orig, f_wasrt

    def update_metrics_from_annotations():
        state["eval_ours"] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        state["eval_wasrt"] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for f_idx, ann in state["frame_annotations"].items():
            if not isinstance(ann, dict):
                continue
            o = ann.get("ours")
            if o in state["eval_ours"]:
                state["eval_ours"][o] += 1
            w = ann.get("wasrt")
            if w in state["eval_wasrt"]:
                state["eval_wasrt"][w] += 1

    def set_frame_tag(f_idx, method, tag):
        k = str(f_idx)
        if k not in state["frame_annotations"]:
            state["frame_annotations"][k] = {"ours": None, "wasrt": None}
        state["frame_annotations"][k][method] = tag
        update_metrics_from_annotations()

    update_metrics_from_annotations()

    while True:
        if current_frame >= total_frames:
            current_frame = total_frames - 1
        if current_frame < 0:
            current_frame = 0
            
        ret, f_orig, f_wasrt = fetch_frames(current_frame)
        if not ret:
            print(f"[!] End of video or error reading frame {current_frame}")
            break
            
        target_h, target_w = 384, 512
        v_orig = cv2.resize(f_orig, (target_w, target_h))
        # Generar Ours EN VIVO sobre el frame original: SOLO horizonte azul y cuadros rojos (SIN verde ni amarillo)
        v_ours, boxes_ours = process_ours(f_orig, target_size=(target_w, target_h))
        if show_masks:
            res_m = detect_obstacles(f_orig, target_size=(256, 192))
            m_a = cv2.resize(res_m.get("mask_agua", np.zeros((192, 256), dtype=np.uint8)), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            m_o = cv2.resize(res_m.get("mask_obstaculos", np.zeros((192, 256), dtype=np.uint8)), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            v_ours[m_a > 0] = cv2.addWeighted(v_ours[m_a > 0], 0.7, np.full_like(v_ours[m_a > 0], (200, 100, 30)), 0.3, 0)
            v_ours[m_o > 0] = [0, 0, 255]
            cv2.putText(v_ours, "FCM Pixel Mask ON", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        v_wasrt = cv2.resize(f_wasrt, (target_w, target_h))
        
        # Label each panel (in English)
        cv2.putText(v_orig, f"Original (Frame {current_frame}/{total_frames})", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(v_ours, "Ours (Fuzzy Logic)", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(v_wasrt, "WaSR-T (Temporal CNN)", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2, cv2.LINE_AA)
        
        ann = state["frame_annotations"].get(str(current_frame)) or {}
        ours_tag = str(ann.get("ours") or "---").upper()
        wasrt_tag = str(ann.get("wasrt") or "---").upper()
        
        cv2.putText(v_ours, f"Status: [{ours_tag}]", (15, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ours_tag in ["TP", "TN"] else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(v_ours, f"Boxes: {len([b for b in boxes_ours if b])}", (15, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(v_wasrt, f"Status: [{wasrt_tag}]", (15, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if wasrt_tag in ["TP", "TN"] else (0, 0, 255), 2, cv2.LINE_AA)

        # Real-time framing algorithm on WaSR-T obstacle mask
        hsv_w = cv2.cvtColor(v_wasrt, cv2.COLOR_BGR2HSV)
        yellow_mask_w = cv2.inRange(hsv_w, np.array([15, 60, 60]), np.array([45, 255, 255]))
        cuadros_wasrt = mask_to_bounding_boxes(yellow_mask_w, min_area=30)
        for c in cuadros_wasrt:
            cv2.rectangle(v_wasrt, (c["x_init"], c["y_init"]), (c["x_end"], c["y_end"]), (0, 0, 255), 2)
            cv2.putText(v_wasrt, f"{c['weight']}px", (c["x_init"], max(15, c["y_init"] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(v_wasrt, f"Boxes: {len(cuadros_wasrt)}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 2, cv2.LINE_AA)

        # Concatenate horizontally
        combined = np.hstack([v_orig, v_ours, v_wasrt])
        
        # Bottom HUD bar with global metrics
        p_o, r_o, f1_o, tot_o = calc_metrics(state["eval_ours"])
        p_w, r_w, f1_w, tot_w = calc_metrics(state["eval_wasrt"])
        
        total_annotated = len([k for k, v in state["frame_annotations"].items() if isinstance(v, dict) and (v.get("ours") or v.get("wasrt"))])
        
        bar = np.zeros((82, combined.shape[1], 3), dtype=np.uint8)
        hud_text1 = f"OURS:   TP={state['eval_ours']['tp']} FP={state['eval_ours']['fp']} FN={state['eval_ours']['fn']} TN={state['eval_ours']['tn']} | Prec={p_o:.3f} Rec={r_o:.3f} F1={f1_o:.3f}"
        hud_text2 = f"WaSR-T: TP={state['eval_wasrt']['tp']} FP={state['eval_wasrt']['fp']} FN={state['eval_wasrt']['fn']} TN={state['eval_wasrt']['tn']} | Prec={p_w:.3f} Rec={r_w:.3f} F1={f1_w:.3f}"
        hud_text3 = f"Frame {current_frame}/{total_frames} (Validados: {total_annotated}) | Salto: {step} | [b]=Ambos TP | [0]=Ambos TN | [c]=Borrar"
        hud_text4 = "[1-4]=Ours | [q,w,e,r]=WaSR-T | [u]=Auto-Ours | [U]=Recalcular Todo | [p]=Mascara | [s]=Guardar | [ESC]=Salir"
        
        cv2.putText(bar, hud_text1, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, hud_text2, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 150, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, hud_text3, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(bar, hud_text4, (15, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        
        display_frame = np.vstack([combined, bar])
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        frame_key = str(current_frame)
            
        # Teclas de salida y guardado
        if key in [27, ord('x')]:  # ESC o x
            state["last_frame"] = current_frame
            state["step"] = step
            update_metrics_from_annotations()
            save_checkpoint(state)
            print("\n[+] Guardando y saliendo...")
            break
        elif key == ord('s'):
            update_metrics_from_annotations()
            save_checkpoint(state)
            tbl = generate_latex_table(state["eval_ours"], state["eval_wasrt"])
            print("\n[+] Checkpoint guardado. Tabla LaTeX generada:\n")
            print(tbl)
            continue
            
        # Modificar salto de frames
        elif key in [ord('+'), ord('=')]:
            step = min(150, step + 5)
            print(f"[i] Salto cambiado a: {step} frames")
            continue
        elif key in [ord('-'), ord('_')]:
            step = max(1, step - 5)
            print(f"[i] Salto cambiado a: {step} frames")
            continue

        # Borrar anotación del frame actual
        elif key == ord('c'):
            if frame_key in state["frame_annotations"]:
                del state["frame_annotations"][frame_key]
                update_metrics_from_annotations()
                print(f"[i] Anotación del frame {current_frame} eliminada.")
            continue

        # Alternar máscara FCM de píxeles
        elif key == ord('p'):
            show_masks = not show_masks
            print(f"[i] Superposición semántica píxel FCM: {'ACTIVADA' if show_masks else 'DESACTIVADA'}")
            continue

        # Auto-clasificación Ours para el frame actual
        elif key == ord('u'):
            w = ann.get("wasrt")
            if w in ["tp", "fn"]: gt = True
            elif w in ["tn", "fp"]: gt = False
            else: gt = (len(boxes_ours) > 0)
            tag = "tp" if (gt and len(boxes_ours) > 0) else "fp" if (not gt and len(boxes_ours) > 0) else "fn" if (gt and len(boxes_ours) == 0) else "tn"
            set_frame_tag(current_frame, "ours", tag)
            print(f"[i] Frame {current_frame} clasificado como {tag.upper()} (Cajas Ours: {len(boxes_ours)})")
            continue

        # Recalcular todo el video anotado con el algoritmo renovado
        elif key == ord('U'):
            recompute_annotations_headless(state)
            update_metrics_from_annotations()
            continue
            
        # Accesos rápidos para ambos
        elif key == ord('b'):  # Ambos detectaron bien
            set_frame_tag(current_frame, "ours", "tp")
            set_frame_tag(current_frame, "wasrt", "tp")
            current_frame += step
        elif key == ord('0'):  # Ambos agua limpia sin obstáculos
            set_frame_tag(current_frame, "ours", "tn")
            set_frame_tag(current_frame, "wasrt", "tn")
            current_frame += step
            
        # Teclas individuales Ours
        elif key == ord('1'):
            set_frame_tag(current_frame, "ours", "tp")
        elif key == ord('2'):
            set_frame_tag(current_frame, "ours", "fp")
        elif key == ord('3'):
            set_frame_tag(current_frame, "ours", "fn")
        elif key == ord('4'):
            set_frame_tag(current_frame, "ours", "tn")
            
        # Teclas individuales WaSR-T
        elif key == ord('q'):
            set_frame_tag(current_frame, "wasrt", "tp")
        elif key == ord('w'):
            set_frame_tag(current_frame, "wasrt", "fp")
        elif key == ord('e'):
            set_frame_tag(current_frame, "wasrt", "fn")
        elif key == ord('r'):
            set_frame_tag(current_frame, "wasrt", "tn")
            
        # Navegación
        elif key in [32, ord('d')]:  # Espacio o d
            current_frame += step
        elif key == ord('a'):  # a
            current_frame -= step

    cap_orig.release()
    cap_wasrt.release()
    cv2.destroyAllWindows()
    
    # Reporte final
    update_metrics_from_annotations()
    save_checkpoint(state)
    latex = generate_latex_table(state["eval_ours"], state["eval_wasrt"])
    print("\n" + "=" * 70)
    print("              RESULTADOS FINALES Y TABLA LATEX")
    print("=" * 70)
    print(latex)
    print(f"[+] Archivo LaTeX guardado en: {LATEX_TABLE_FILE}")
    print(f"[+] Checkpoint JSON guardado en: {CHECKPOINT_FILE}")

if __name__ == "__main__":
    main()
