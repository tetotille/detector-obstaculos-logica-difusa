#image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/barco.jpg"
import cv2
import numpy as np

def detect_horizon(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos de color para el cielo y el mar
    lower_sky = np.array([90, 50, 50])
    upper_sky = np.array([130, 255, 255])
    lower_sea = np.array([90, 50, 20])
    upper_sea = np.array([130, 255, 120])

    # Crear máscaras para el cielo y el mar
    sky_mask = cv2.inRange(hsv_image, lower_sky, upper_sky)
    sea_mask = cv2.inRange(hsv_image, lower_sea, upper_sea)

    # Encontrar contornos en las máscaras
    sky_contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sea_contours, _ = cv2.findContours(sea_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar la línea de horizonte
    horizon_y = None
    if sky_contours and sea_contours:
        sky_bottom = max(sky_contours, key=cv2.contourArea)[0][0][1]
        sea_top = min(sea_contours, key=cv2.contourArea)[0][0][1]
        horizon_y = (sky_bottom + sea_top) // 2

    # Dibujar la línea de horizonte
    if horizon_y is not None:
        cv2.line(image, (0, horizon_y), (image.shape[1], horizon_y), (0, 255, 0), thickness=2)

    # Mostrar la imagen con la línea de horizonte detectada
    resized_image = cv2.resize(image, (300, 300))
    cv2.imshow('Horizon Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Ruta a la imagen
image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/barco.jpg"
detect_horizon(image_path)
