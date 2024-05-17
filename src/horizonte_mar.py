import cv2
import numpy as np

image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/barco.jpg"
image = cv2.imread(image_path)

def detect_horizon(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta del archivo y asegúrate de que el archivo exista.")

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

    height, width = sky_mask.shape
    sky_bottom = 0
    sea_top = height - 1

    # Encontrar el límite inferior del cielo desde arriba
    for y in range(height):
        if cv2.countNonZero(sky_mask[y, :]) > width * 0.7:  # Si más del 70% de la fila es cielo
            sky_bottom = y

    # Encontrar el límite superior del mar desde abajo
    for y in range(height - 1, -1, -1):
        if cv2.countNonZero(sea_mask[y, :]) > width * 0.7:  # Si más del 70% de la fila es mar
            sea_top = y
            break

    # Calcular la línea de horizonte como la línea media entre el límite inferior del cielo y el límite superior del mar
    horizon_y = (sky_bottom + sea_top) // 2

    # Dibujar la línea de horizonte detectada
    if horizon_y is not None:
        cv2.line(image, (0, horizon_y), (width, horizon_y), (0, 255, 0), thickness=2)

    # Mostrar la imagen con el horizonte detectado
    resized_image = cv2.resize(image, (300, 300))
    cv2.imshow('Horizon Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_horizon(image_path)

