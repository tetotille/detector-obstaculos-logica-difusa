from os.path import abspath,dirname,join
from sys import argv
import cv2
import numpy as np

def adjust_contrast_brightness(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def detect_objects(image):
    # Ajustar el contraste y brillo de la imagen
    adjusted_image = adjust_contrast_brightness(image, alpha=1, beta=20)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro de desenfoque gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(adjusted_image, (5, 5), 0)
    
    # Aplicar detección de bordes usando Canny
    edges = cv2.Canny(blurred, 100, 200)
    
    # Dilatar los bordes para cerrar gaps entre segmentos
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Encontrar contornos en la imagen de bordes dilatados
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una copia de la imagen para dibujar los contornos
    result_image = adjusted_image.copy()
    
    # Dibujar rectángulos o polígonos alrededor de los objetos detectados
    for contour in contours:
        if cv2.contourArea(contour) > 2000:  # Filtrar contornos pequeños
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Si el contorno tiene 4 lados, dibujar un rectángulo
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:  # De lo contrario, dibujar un polígono
                cv2.polylines(result_image, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return result_image

# Ruta de la imagen
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))),"img/barco.jpg")

##############################################################################

print(filename)

# Cargar la imagen
image = cv2.imread(filename)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("No se pudo cargar la imagen.")
else:
    # Detectar objetos en el agua
    result = detect_objects(image)
    
    # Mostrar la imagen con los objetos detectados
    resized_image = cv2.resize(result, (600, 600))
    cv2.imshow("Detección de Objetos en el Agua", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
