import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2

from src.main_gpu import get_video_stream

def main():

    # Abrir la webcam (0 es generalmente la cámara principal)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        exit()

    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame.")
            break

        # Mostrar el frame en una ventana
        cv2.imshow('Webcam', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame = get_video_stream()

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

