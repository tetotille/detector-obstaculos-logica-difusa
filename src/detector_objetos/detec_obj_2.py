from os.path import abspath, dirname, join
from sys import argv
import cv2
import numpy as np

def adjust_contrast_brightness(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def detect_objects_median_dog(image):
    # Ajustar el contraste y brillo de la imagen
    adjusted_image = adjust_contrast_brightness(image, alpha=1, beta=20)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro de Mediana para reducir el ruido
    median_filtered = cv2.medianBlur(gray, 5)
    
    # Aplicar Diferencia de Gaussianas (DoG)
    blur1 = cv2.GaussianBlur(median_filtered, (3, 3), 0)
    blur2 = cv2.GaussianBlur(median_filtered, (17, 17), 0)

    dog = cv2.absdiff(blur1, blur2)
    
    # Binarizar la imagen
    _, binary_map = cv2.threshold(dog, 18, 255, cv2.THRESH_BINARY)
    
    # Aplicar operaciones morfológicas para limpiar la imagen binaria
    kernel = np.ones((5, 5), np.uint8)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una copia de la imagen para dibujar los contornos
    result_image = adjusted_image.copy()
    
    # Dibujar contornos alrededor de los objetos detectados
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filtrar contornos pequeños
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result_image

# Ruta de la imagen
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/sintitulo.jpg")

print(filename)

# Cargar la imagen
image = cv2.imread(filename)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("No se pudo cargar la imagen.")
else:
    # Detectar objetos en el agua usando filtro de Mediana y Diferencia de Gaussianas
    result = detect_objects_median_dog(image)
    
    # Mostrar la imagen con los objetos detectados
    resized_image = cv2.resize(result, (300, 300))
    cv2.imshow("Detección de Objetos en el Agua", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
