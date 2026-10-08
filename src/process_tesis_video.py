import sys
import os
import time
import cv2

# Añadir el directorio raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline_fuzzy_detector import detect_obstacles


def process_video(video_path):
    print(f"Iniciando procesamiento de video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return

    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    output_path = "main_output/tesis_procesado.mp4"
    os.makedirs("main_output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (256, 192))

    frame_count = 0
    display_available = "DISPLAY" in os.environ
    if display_available:
        print("Display detectado. Mostrando procesamiento en tiempo real...")
    else:
        print("No se detectó entorno gráfico (DISPLAY). El procesamiento continuará sin visualización.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        t0 = time.time()

        res = detect_obstacles(frame, target_size=(256, 192))

        # Cuadro redimensionado para dibujo
        vis_frame = cv2.resize(frame, (256, 192))
        y_h = res["horizon"]["ajuste"]
        cv2.line(vis_frame, (0, y_h), (256, y_h), (255, 100, 0), 2)

        # Dibujar obstáculos confirmados por consenso FCM e intersección
        for b in res["confirmed_boxes"]:
            x1, y1 = int(b["x_init"]), int(b["y_init"])
            x2, y2 = int(b["x_end"]), int(b["y_end"])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Score:{b.get('fuzzy_union', 0.0):.1f}", 
                        (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Mostrar en tiempo real si hay display
        if display_available:
            cv2.imshow("Procesamiento en Tiempo Real", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Escribir frame al video de salida
        out.write(vis_frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count} procesado. Tiempo: {time.time() - t0:.3f}s (obs={len(res['confirmed_boxes'])})")

    cap.release()
    out.release()
    if display_available:
        cv2.destroyAllWindows()
    print(f"Procesamiento finalizado. Video guardado en: {output_path}")


if __name__ == "__main__":
    video_file = "/home/tetotille/Proyectos/detector-obstaculos-logica-difusa/assets/videos/tesis.mp4"
    process_video(video_file)
