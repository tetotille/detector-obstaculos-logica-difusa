import cv2
import numpy as np
import json

def calcular_angulo_rotacion(imagen_path):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("Error al cargar la imagen")
        return

    # Redimensionar la imagen para acelerar el procesamiento
    imagen_redimensionada = cv2.resize(imagen, (800, int(imagen.shape[0] * 800 / imagen.shape[1])))

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque para reducir el ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # Aplicar el detector de bordes de Canny
    bordes = cv2.Canny(gris, 50, 150)

    # Detectar líneas usando la Transformada de Hough
    lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 200)

    if lineas is None:
        print("No se detectaron líneas en la imagen.")
        return

    # Filtrar las líneas horizontales y calcular el ángulo promedio
    angulos = []
    for linea in lineas:
        rho, theta = linea[0]
        angulo = np.degrees(theta)
        if 80 < angulo < 100 or -100 < angulo < -80:  # Filtro para líneas horizontales
            angulos.append(angulo - 90)

    if len(angulos) == 0:
        print("No se detectaron líneas horizontales en la imagen.")
        return

    # Calcular el ángulo de rotación promedio
    angulo_rotacion = np.mean(angulos)

    print(f"Ángulo de rotación estimado: {angulo_rotacion:.2f} grados")

    # Devolver el ángulo de rotación estimado
    return angulo_rotacion

# Ruta de la imagen
imagen_path = json.load(open("config.json"))["img_path"] + "ruta-vista-inclinada-que-cruza-horizonte.jpg"

# Calcular el ángulo de rotación
angulo_rotacion = calcular_angulo_rotacion(imagen_path)

if angulo_rotacion is not None:
    if abs(angulo_rotacion) > 5:  # Umbral para considerar la imagen rotada
        print("La cámara parece estar rotada.")
    else:
        print("La cámara parece estar correctamente alineada.")











