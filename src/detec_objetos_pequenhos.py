from os.path import abspath, dirname, join
from sys import argv
import cv2
import numpy as np

def adjust_contrast_brightness(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def detect_objects_hough(image):
    # Ajustar el contraste y brillo de la imagen
    adjusted_image = adjust_contrast_brightness(image, alpha=1.5, beta=20)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar la Transformada de Hough para detectar círculos
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=70, param2=30, minRadius=5, maxRadius=50)
    
    # Crear una copia de la imagen para dibujar los círculos
    result_image = adjusted_image.copy()
    
    # Dibujar los círculos detectados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(result_image, (x - r, y - r), (x + r, y + r), (0, 128, 255), 2)
    
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
    # Detectar objetos en el agua con Transformada de Hough
    result = detect_objects_hough(image)
    
    # Mostrar la imagen con los objetos detectados
    resized_image = cv2.resize(result, (300, 300))
    cv2.imshow("Detección de Objetos en el Agua", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detección de círculos usando la Transformada de Hough
    #circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=70, param2=30, minRadius=5, maxRadius=50)
    
