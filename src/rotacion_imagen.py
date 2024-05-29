import cv2
import json
import numpy as np

def calcular_angulo_rotacion(imagen_path):
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("Error al cargar la imagen")
        return

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150, apertureSize=3)

    # Detectar líneas usando la Transformada de Hough 
    lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 200)

    angulos = []
    if lineas is not None:
        for linea in lineas:
            rho, theta = linea[0]
            # Convertir el ángulo de la línea de radianes a grados
            angulo = np.degrees(theta)
            # Normalizar el ángulo a un rango de -90 a 90 grados
            if angulo > 90:
                angulo -= 180
            angulos.append(angulo)

    if len(angulos) == 0:
        print("No se detectaron líneas significativas en la imagen.")
        return

    # Calcular el ángulo de rotación promedio
    angulo_rotacion = np.mean(angulos)

    print(f"Ángulo de rotación promedio: {angulo_rotacion:.2f} grados")

    # Devolver el ángulo de rotación promedio
    return angulo_rotacion

# Ruta de la imagen
imagen_path = json.load(open("config.json"))["img_path"] + "normal2.webp"

# Calcular el ángulo de rotación
angulo_rotacion = calcular_angulo_rotacion(imagen_path)

if angulo_rotacion is not None:
    if abs(angulo_rotacion) > 10:  # Umbral para considerar la imagen rotada
        print("La cámara parece estar rotada.")
    else:
        print("La cámara parece estar correctamente alineada.")
