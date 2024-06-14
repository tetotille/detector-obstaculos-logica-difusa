import cv2
import numpy as np
import json
# De algunos se tiene que reducir el factor k, de algunos se tiene que aumentar el brillo, o sea los factores
# alfa y beta.

def adjust_contrast_brightness(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#k=5 para el amanecer
#K=2 por defecto
def detect_objects(image, k=5):
    # Ajustar el contraste y brillo de la imagen
    adjusted_image = adjust_contrast_brightness(image, alpha=2, beta=20)
    #alfa = 2 para el amanecer

    # Convertir la imagen de BGR a espacio de color LAB
    lab_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2LAB)
    
    # Remodelar la imagen en una matriz de píxeles
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Definir criterios de K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Aplicar K-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir los centros a valores de 8 bits
    centers = np.uint8(centers)
    
    # Mapear los labels a los centros para crear la imagen segmentada
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Convertir la imagen segmentada a escala de grises
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro de textura (por ejemplo, filtro de Laplaciano) para resaltar las texturas
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.uint8(np.absolute(texture))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Umbralizar la imagen de textura para obtener una máscara binaria
    _, binary_mask = cv2.threshold(texture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Encontrar contornos en la máscara binaria
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una copia de la imagen para dibujar los rectángulos
    result_image = image.copy()
    
    # Dibujar rectángulos alrededor de los objetos detectados
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filtrar contornos pequeños
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result_image

# Ru+ta de la imagen
image_path = json.load(open("config.json"))["img_path"] + "sintitulo.jpg"

# Cargar la imagen
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("No se pudo cargar la imagen.")
else:
    # Detectar objetos en el agua
    result = detect_objects(image)
    
    # Mostrar la imagen con los objetos detectados
    resized_image = cv2.resize(result, (300,300))
    cv2.imshow("Detección de Objetos en el Agua", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
