import cv2
import os
from sys import argv
from os.path import abspath,dirname,join

#cap = cv2.VideoCapture(f"{json.load(open.('config.json'))['video_path']}LANCHA_RC.mp4")
if len(argv) <= 1: raise(NameError("You have to enter a file name as an argument"))
if os.path.dirname(argv[1]):
    image = cv2.imread(argv[1])
else:
    img_path = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
    image = cv2.imread(img_path)


# Especificar el nombre del archivo de video dentro de la ruta seleccionada

cap=cv2.VideoCapture(os.path.join(ruta_seleccionada, "video1.mp4"))
# Leer el primer fotograma
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # Leer el fotograma actual
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma actual a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular la diferencia absoluta entre el fotograma actual y el anterior
    frame_diff = cv2.absdiff(prev_gray, gray)

    # Aplicar un umbral para resaltar las regiones con cambios significativos
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la máscara binaria
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar rectángulos alrededor de los contornos encontrados
    for contour in contours:
        # Calcular el área del contorno
        area = cv2.contourArea(contour)
        # Si el área es lo suficientemente grande, dibujar un rectángulo
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el fotograma actual y la máscara de diferencia
    cv2.imshow("Frame", frame)
    cv2.imshow("Difference", thresh)

    # Actualizar el fotograma anterior con el actual para el siguiente ciclo
    prev_gray = gray.copy()

    # Detener el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

