import cv2
import threading
import time
import prueba_torch

class VideoHandler:
    def __init__(self, path_video):
        self.video = cv2.VideoCapture(path_video)
        self.current_frame = None
        self.frame_ready = threading.Event()
        self.stop_flag = False

    def start(self):
        """Inicia el hilo de captura de video."""
        thread = threading.Thread(target=self._capture_frames)
        thread.start()

    def _capture_frames(self):
        """Captura frames del video en tiempo real."""
        while not self.stop_flag:
            # Leer un frame
            ret, frame = self.video.read()

            # Si no hay más frames, detener el bucle
            if not ret:
                self.stop_flag = True
                self.frame_ready.set()  # Liberar el frame final
                break

            # Almacenar el frame actual
            self.current_frame = frame
            self.frame_ready.set()  # Señal para que el frame esté listo

            # Esperar a que el frame sea procesado antes de capturar otro
            time.sleep(0.03)  # Ajusta el tiempo según la velocidad deseada

    def get_frame(self):
        """Obtiene el frame actual cuando esté listo."""
        self.frame_ready.wait()  # Espera hasta que haya un frame listo
        self.frame_ready.clear()  # Marca el frame como "en proceso"
        return self.current_frame

    def stop(self):
        """Detiene la captura de video y libera los recursos."""
        self.stop_flag = True
        self.video.release()


def menu_principal(path_video):
    # Crear un objeto de manejo de video
    video_handler = VideoHandler(path_video)

    # Iniciar el hilo de captura de video
    video_handler.start()

    try:
        while not video_handler.stop_flag:
            # Obtener el siguiente frame para su procesamiento
            frame = video_handler.get_frame()

            # Procesar el frame
            if frame is not None:
                prueba_torch.procesar_imagen(frame)
            else:
                break

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")

    finally:
        # Detener la captura de video y cerrar ventanas
        video_handler.stop()
        cv2.destroyAllWindows()

# Llamar a la función principal con la ruta del video
menu_principal("ruta/al/video.mp4")
