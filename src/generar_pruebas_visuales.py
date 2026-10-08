import os
# Asegurar compatibilidad gráfica Qt
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline_fuzzy_detector import detect_obstacles
from src.utils.utils import read_image

OUTPUT_DIR = project_root / "main_output/pruebas_visuales"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar caché de WaSR-T si existe para comparación
CACHE_DIR = project_root / "main_output/wasrt_cache"

def procesar_y_crear_panel(img_bgr, titulo, wasrt_mask=None):
    """
    Genera un panel comparativo de 4 vistas:
    [1. Original + Horizonte] | [2. Segmentación Semántica FCM] | [3. Cajas Finales Encuadradas] | [4. Comparativa WaSR-T]
    """
    res = detect_obstacles(img_bgr, target_size=(256, 192))
    
    w, h = 256, 192
    img_res = cv2.resize(img_bgr, (w, h))
    y_h = res["horizon"]["ajuste"]
    
    # 1. Panel Original con Horizonte
    p1 = img_res.copy()
    cv2.line(p1, (0, y_h), (w, y_h), (255, 100, 0), 2)
    cv2.putText(p1, "1. Horizonte Estimado", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # 2. Panel de Segmentación Semántica a Nivel de Píxel (Agua vs Obstáculo)
    # Se superpone sobre el área de agua: Agua en azul translúcido, Obstáculos en rojo sólido
    p2 = img_res.copy()
    m_agua = res.get("mask_agua")
    m_obs = res.get("mask_obstaculos")
    
    if m_agua is not None and m_obs is not None:
        # Pintar agua en azul suave
        p2[m_agua > 0] = cv2.addWeighted(p2[m_agua > 0], 0.6, np.full_like(p2[m_agua > 0], (200, 100, 30)), 0.4, 0)
        # Pintar obstáculos en rojo vivo a nivel de píxel
        p2[m_obs > 0] = [0, 0, 255]
        
    cv2.line(p2, (0, y_h), (w, y_h), (255, 100, 0), 1)
    cv2.putText(p2, "2. Segmentacion Pixel FCM", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(p2, "Azul=Agua | Rojo=Obstaculo", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # 3. Panel de Cajas Finales Encuadradas (Post-Segmentación)
    p3 = img_res.copy()
    cv2.line(p3, (0, y_h), (w, y_h), (255, 100, 0), 1)
    boxes = res["confirmed_boxes"]
    for b in boxes:
        x1, y1 = int(b["x_init"]), int(b["y_init"])
        x2, y2 = int(b["x_end"]), int(b["y_end"])
        xc, yc = int(b.get("x_centroid", (x1+x2)//2)), int(b.get("y_centroid", (y1+y2)//2))
        cv2.rectangle(p3, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(p3, (xc, yc), 3, (0, 0, 255), -1)
        cv2.putText(p3, f"Score:{b.get('fuzzy_union', 0):.1f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    status_str = f"3. Cajas ({len(boxes)} obj)" if len(boxes) > 0 else "3. Agua Limpia (0 obj)"
    color_status = (0, 255, 0) if len(boxes) > 0 else (200, 200, 200)
    cv2.putText(p3, status_str, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_status, 1)

    # 4. Panel de Referencia (WaSR-T o Comparativa)
    p4 = img_res.copy()
    if wasrt_mask is not None:
        p4[wasrt_mask > 0] = [0, 0, 255] # Obstáculo WaSR-T en rojo
        cv2.line(p4, (0, y_h), (w, y_h), (255, 100, 0), 1)
        cv2.putText(p4, "4. Referencia WaSR-T", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    else:
        # Si no hay máscara WaSR-T (ej. imagen estática), mostrar diferencia/residuos
        cv2.line(p4, (0, y_h), (w, y_h), (255, 100, 0), 1)
        cv2.putText(p4, "4. Deteccion Confirmada", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for b in boxes:
            cv2.rectangle(p4, (b["x_init"], b["y_init"]), (b["x_end"], b["y_end"]), (0, 255, 0), 2)

    # Encabezado del título
    banner = np.full((32, w * 4, 3), (35, 35, 35), dtype=np.uint8)
    cv2.putText(banner, f"PRUEBA: {titulo}  |  Obstaculo Detectado: {'SI' if res['has_obstacle'] else 'NO'}  |  Cajas Confirmadas: {len(boxes)}",
                (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if res['has_obstacle'] else (200, 200, 200), 1)

    grid = np.hstack([p1, p2, p3, p4])
    full = np.vstack([banner, grid])
    return full

def main():
    print("[+] Generando pruebas visuales con la nueva arquitectura...")
    
    # 1. Probar en cuadros representativos del video
    cap = cv2.VideoCapture(str(project_root / "assets/videos/tesis.mp4"))
    video_frames = [
        (50, "Video Frame 50 (Obstaculo distante nublado)"),
        (100, "Video Frame 100 (Obstaculo lejano nublado)"),
        (200, "Video Frame 200 (Obstaculo en agua)"),
        (300, "Video Frame 300 (Obstaculo en transito)"),
        (500, "Video Frame 500 (Boya/Lancha en navegacion)"),
        (1000, "Video Frame 1000 (Agua limpia sin obstaculos)"),
        (1400, "Video Frame 1400 (Obstaculo mediano)"),
        (1600, "Video Frame 1600 (Retorno de lancha)")
    ]
    
    saved_paths = []
    for f_idx, titulo in video_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Cargar máscara WaSR-T si está disponible
        wasrt_mask = None
        cache_f = CACHE_DIR / f"frame_{f_idx:04d}.npz"
        if cache_f.exists():
            d = np.load(cache_f)
            wasrt_mask = d.get("obs_water_mask")
            
        panel = procesar_y_crear_panel(frame, titulo, wasrt_mask=wasrt_mask)
        out_file = OUTPUT_DIR / f"prueba_video_frame_{f_idx:04d}.png"
        cv2.imwrite(str(out_file), panel)
        saved_paths.append(out_file)
        print(f"  [+] Generado: {out_file.name}")
        
    cap.release()
    
    # 2. Probar en imágenes fijas emblemáticas del repositorio
    static_images = [
        ("assets/images/barco.jpg", "Imagen Estática: barco.jpg"),
        ("assets/images/barco-ypa.jpeg", "Imagen Estática: barco-ypa.jpeg"),
        ("assets/images/lago-ypacarai (1).jpg", "Imagen Estática: lago-ypacarai.jpg")
    ]
    
    for rel_path, titulo in static_images:
        img_path = project_root / rel_path
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                panel = procesar_y_crear_panel(img, titulo)
                safe_name = rel_path.split("/")[-1].replace(" ", "_").replace("(", "").replace(")", "")
                out_file = OUTPUT_DIR / f"prueba_{safe_name}.png"
                cv2.imwrite(str(out_file), panel)
                saved_paths.append(out_file)
                print(f"  [+] Generado: {out_file.name}")
                
    print(f"\n[+] Total de {len(saved_paths)} imágenes de prueba generadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
