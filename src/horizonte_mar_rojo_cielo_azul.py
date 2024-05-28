import cv2
import numpy as np

def detectar_horizonte(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussian para reducir el ruido
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calcular el gradiente de la imagen
    gradient = np.gradient(blurred_image, axis=0)

    # Aplicar una media móvil para suavizar el gradiente
    smoothed_gradient = np.convolve(np.mean(gradient, axis=1), np.ones(15)/15, mode='same')

    # Encontrar el punto de transición más significativo
    transition_index = np.argmax(np.abs(smoothed_gradient))

    # Dibujar la línea de horizonte detectada
    if transition_index is not None:
        cv2.line(image, (0, transition_index), (image.shape[1], transition_index), (0, 255, 0), thickness=2)

    return image, transition_index

def detect_horizon(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta del archivo y asegúrate de que el archivo exista.")

    # Detectar el horizonte
    horizonte_image, horizon_y = detectar_horizonte(image)

    # Mostrar la imagen con el horizonte detectado
    resized_image = cv2.resize(horizonte_image, (600, 600))  # Aumentar el tamaño para mejor visualización
    cv2.imshow('Horizon Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta a la imagen
image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/ruta-vista-inclinada-que-cruza-horizonte.jpg"
detect_horizon(image_path)



