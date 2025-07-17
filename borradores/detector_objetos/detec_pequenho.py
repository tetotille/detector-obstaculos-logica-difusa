from os.path import abspath, dirname, join
from sys import argv
import cv2
import numpy as np

def adjust_contrast_brightness(image, alpha=1, beta=20):
    # Ajuste de contraste y brillo
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def filter_predominant_color(image):
    height, width, _ = image.shape
    bottom_half = image[height//2:, :, :]

    hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    predominant_color = np.unravel_index(np.argmax(hist), hist.shape)
    lower_bound = np.array([predominant_color[0] - 10, 50, 50])
    upper_bound = np.array([predominant_color[0] + 10, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_full = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)

    return cv2.bitwise_and(image, image, mask=~mask_full)

def detect_objects(image):
    adjusted_image = adjust_contrast_brightness(image)  # Ajuste de contraste y brillo
    filtered_image = filter_predominant_color(adjusted_image)  # Filtro de color predominante
    
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Tamaño del kernel
    edges = cv2.Canny(filtered_image, 400, 800)  # Umbrales del detector de bordes (ajustados)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = adjusted_image.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Área mínima del contorno (ajustada)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.3 < circularity < 20:  # Rango de circularidad
                x, y, w, h = cv2.boundingRect(contour)
                if w*h < 10000:  # Área máxima del objeto
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
    # Detectar objetos en el agua filtrando el color predominante en la parte inferior
    result = detect_objects(image)
    
    # Mostrar la imagen con los objetos detectados
    resized_image = cv2.resize(result, (300, 300))
    cv2.imshow("Detección de Objetos en el Agua", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
