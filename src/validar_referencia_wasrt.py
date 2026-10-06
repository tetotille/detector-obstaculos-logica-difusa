import os
# Asegurar compatibilidad con Wayland / XWayland para Qt en OpenCV
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import json
import csv
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline_fuzzy_detector import detect_obstacles

OUTPUT_DIR = project_root / "main_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_JSON = OUTPUT_DIR / "wasrt_reference_validation.json"
CHECKPOINT_CSV = OUTPUT_DIR / "wasrt_reference_validation.csv"

VIDEO_ORIGINAL = project_root / "assets/videos/tesis.mp4"
VIDEO_WASRT = OUTPUT_DIR / "tesis_wasrt_cpu.mp4"


def load_checkpoint():
    if CHECKPOINT_JSON.exists():
        try:
            with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[+] Checkpoint cargado desde: {CHECKPOINT_JSON}")
                return data
        except Exception as e:
            print(f"[!] Error leyendo checkpoint ({e}), creando nuevo.")
    
    return {
        "last_frame": 0,
        "step": 10,
        "annotations": {}  # frame_idx (str) -> bool (True si sirve como ref, False si no)
    }


def save_checkpoint(state):
    # 1. Guardar JSON
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        
    # 2. Guardar CSV ordenado
    anns = state.get("annotations", {})
    sorted_frames = sorted([int(k) for k in anns.keys()])
    
    with open(CHECKPOINT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "wasrt_sirve_referencia"])
        for f_idx in sorted_frames:
            writer.writerow([f_idx, "SI" if anns[str(f_idx)] else "NO"])


def main():
    print("=" * 75)
    print("  VALIDADOR INTERACTIVO DE REFERENCIA: ¿WaSR-T SIRVE EN ESTE FRAME?")
    print("=" * 75)
    
    if not VIDEO_ORIGINAL.exists():
        print(f"[!] ERROR: No se encontró el video original en: {VIDEO_ORIGINAL}")
        return
    if not VIDEO_WASRT.exists():
        print(f"[!] ERROR: No se encontró el video de WaSR-T en: {VIDEO_WASRT}")
        return

    cap_orig = cv2.VideoCapture(str(VIDEO_ORIGINAL))
    cap_wasrt = cv2.VideoCapture(str(VIDEO_WASRT))
    
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    
    state = load_checkpoint()
    current_frame = state.get("last_frame", 0)
    step = state.get("step", 10)
    annotations = state.get("annotations", {})
    
    window_name = "Validador de Referencia WaSR-T [Tesis Lake USV]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1500, 520)

    print("\n[CONTROLES DEL TECLADO]:")
    print("  -------------------------------------------------------------")
    print("  [y] / [v] / [1] : SÍ - WaSR-T SIRVE como referencia (Avanza)")
    print("  [n] / [x] / [0] : NO - WaSR-T NO SIRVE / Tiene errores (Avanza)")
    print("  [ESPACIO] / [d] : Siguiente frame (sin cambiar etiqueta)")
    print("  [a]             : Frame anterior")
    print("  [+] / [-]       : Aumentar / Disminuir salto (actual: {})".format(step))
    print("  [c]             : Borrar etiqueta de este frame")
    print("  [s]             : Guardar inmediatamente")
    print("  [q] / [ESC]     : Guardar y salir")
    print("  -------------------------------------------------------------\n")

    panel_size = (480, 360)
    
    while True:
        current_frame = max(0, min(current_frame, total_frames - 1))
        state["last_frame"] = current_frame
        state["step"] = step

        # Leer frames
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        cap_wasrt.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        ret1, f_orig = cap_orig.read()
        ret2, f_wasrt = cap_wasrt.read()
        
        if not ret1 or not ret2:
            print(f"[!] Fin del video o error leyendo frame {current_frame}")
            break

        # Redimensionar para los paneles
        p_orig = cv2.resize(f_orig, panel_size)
        p_wasrt = cv2.resize(f_wasrt, panel_size)
        
        # Panel 3: Ejecutar nuestro método difuso para comparación visual
        det_fuzzy = detect_obstacles(f_orig, target_size=(256, 192))
        p_fuzzy = cv2.resize(f_orig, panel_size)
        
        # Dibujar horizonte y cajas difusas
        h_info = det_fuzzy.get("horizon", {})
        h_y = int(h_info.get("ajuste", 0) * (panel_size[1] / 192.0))
        cv2.line(p_fuzzy, (0, h_y), (panel_size[0], h_y), (255, 0, 0), 2)
        
        scale_x = panel_size[0] / 256.0
        scale_y = panel_size[1] / 192.0
        for box in det_fuzzy.get("confirmed_boxes", []):
            x1 = int(box["x_init"] * scale_x)
            x2 = int(box["x_end"] * scale_x)
            y1 = int(box["y_init"] * scale_y)
            y2 = int(box["y_end"] * scale_y)
            cv2.rectangle(p_fuzzy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(p_fuzzy, f"Ours:{box.get('weight', 0)}px", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Encabezados de paneles
        cv2.putText(p_orig, "1. ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(p_wasrt, "2. WaSR-T (REFERENCIA)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(p_fuzzy, "3. NUESTRO (DIFUSO)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Unir los 3 paneles
        canvas = np.hstack([p_orig, p_wasrt, p_fuzzy])
        
        # Estado actual de anotación
        f_key = str(current_frame)
        status_text = "PENDIENTE DE REVISAR"
        status_color = (180, 180, 180) # Gris
        bg_bar_color = (40, 40, 40)
        
        if f_key in annotations:
            if annotations[f_key] is True:
                status_text = "WaSR-T SIRVE COMO REFERENCIA: [ SÍ ]"
                status_color = (0, 255, 0) # Verde
                bg_bar_color = (0, 70, 0)
            elif annotations[f_key] is False:
                status_text = "WaSR-T NO SIRVE (TIENE ERRORES): [ NO ]"
                status_color = (0, 0, 255) # Rojo
                bg_bar_color = (0, 0, 70)

        # Estadísticas acumuladas
        num_valid = sum(1 for v in annotations.values() if v is True)
        num_invalid = sum(1 for v in annotations.values() if v is False)
        total_rev = len(annotations)

        # Barra superior de información
        top_bar = np.full((70, canvas.shape[1], 3), bg_bar_color, dtype=np.uint8)
        
        line1 = f"Frame: {current_frame} / {total_frames}  |  Salto: {step} frames  |  Revisados: {total_rev} (Validos: {num_valid}, Invalidos: {num_invalid})"
        cv2.putText(top_bar, line1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        
        cv2.putText(top_bar, f"ESTADO: {status_text}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Barra inferior de ayuda
        bot_bar = np.full((35, canvas.shape[1], 3), (25, 25, 25), dtype=np.uint8)
        help_text = "[y]/[1]=SIRVE  |  [n]/[0]=NO SIRVE  |  [ESPACIO]/[d]=Siguiente  |  [a]=Anterior  |  [s]=Guardar  |  [q]=Salir"
        cv2.putText(bot_bar, help_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        
        full_display = np.vstack([top_bar, canvas, bot_bar])
        cv2.imshow(window_name, full_display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key in [ord('q'), 27]: # 'q' o ESC
            print("\n[+] Guardando checkpoint y saliendo...")
            save_checkpoint(state)
            break
            
        elif key in [ord('y'), ord('v'), ord('1')]:
            annotations[f_key] = True
            save_checkpoint(state)
            print(f"  Frame {current_frame:4d}: WaSR-T [SÍ] sirve como referencia.")
            current_frame += step
            
        elif key in [ord('n'), ord('x'), ord('0')]:
            annotations[f_key] = False
            save_checkpoint(state)
            print(f"  Frame {current_frame:4d}: WaSR-T [NO] sirve como referencia.")
            current_frame += step
            
        elif key in [ord(' '), ord('d'), 83]: # Espacio, 'd', o Flecha derecha
            current_frame += step
            
        elif key in [ord('a'), 81]: # 'a' o Flecha izquierda
            current_frame -= step
            
        elif key in [ord('+'), ord('=')]:
            step = min(50, step + 5)
            print(f"  [>] Salto aumentado a: {step} frames")
            
        elif key in [ord('-'), ord('_')]:
            step = max(1, step - 5)
            print(f"  [>] Salto reducido a: {step} frames")
            
        elif key == ord('c'):
            if f_key in annotations:
                del annotations[f_key]
                save_checkpoint(state)
                print(f"  [x] Etiqueta eliminada para frame {current_frame}")
                
        elif key == ord('s'):
            save_checkpoint(state)
            print(f"  [+] Checkpoint guardado manualmente ({total_rev} frames anotados).")

    cap_orig.release()
    cap_wasrt.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 65)
    print("  RESUMEN DE VALIDACIÓN")
    print("=" * 65)
    print(f"Total frames anotados:    {len(annotations)}")
    print(f"WaSR-T Válido (Sirve):    {sum(1 for v in annotations.values() if v is True)}")
    print(f"WaSR-T Inválido (Falla):  {sum(1 for v in annotations.values() if v is False)}")
    print(f"Archivos guardados:")
    print(f"  - {CHECKPOINT_JSON}")
    print(f"  - {CHECKPOINT_CSV}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
