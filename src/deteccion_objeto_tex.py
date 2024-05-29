import cv2
<<<<<<< HEAD
=======
import json
>>>>>>> 9521a56d5c2466d084a4c3f7a5f6a7895f55e4a7
import numpy as np

def detect_objects(image):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar ecualización del histograma para mejorar el contraste
    gray = cv2.equalizeHist(gray)
    
    # Aplicar un filtro de desenfoque gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar detección de bordes usando Canny
    edges = cv2.Canny(blurred, 50, 500)
    
    # Dilatar los bordes para cerrar gaps entre segmentos
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Encontrar contornos en la imagen de bordes dilatados
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una máscara para la segmentación
    mask = np.zeros_like(gray)
    
    # Dibujar los contornos en la máscara
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar contornos pequeños
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Aplicar la transformación de distancia
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normalizar la transformación de distancia
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Umbralizar la transformación de distancia
    _, sure_fg = cv2.threshold(dist_transform, 0.5, 1.0, cv2.THRESH_BINARY)
    
    # Convertir a uint8
    sure_fg = np.uint8(sure_fg)
    
    # Encontrar el fondo seguro
    sure_bg = cv2.dilate(mask, kernel, iterations=1)
    
    # Restar el fondo seguro del foreground seguro para encontrar regiones desconocidas
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Etiquetar marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Aumentar todos los marcadores en 1, para que el fondo sea 1 en lugar de 0
    markers = markers + 1
    
    # Marcar las regiones desconocidas con 0
    markers[unknown == 255] = 0
    
    # Aplicar el algoritmo de Watershed
    markers = cv2.watershed(image, markers)
    
    # Crear una copia de la imagen para dibujar los rectángulos
    result = image.copy()
    
    # Mostrar imagen original
    cv2.imshow("Original", image)

# Mostrar imagen en escala de grises
    cv2.imshow("Grayscale", gray)

# Mostrar imagen desenfocada
    cv2.imshow("Blurred", blurred)

# Mostrar bordes detectados
    cv2.imshow("Edges", edges)

# Mostrar bordes dilatados
    cv2.imshow("Dilated", dilated)

# Mostrar máscara de contornos
    cv2.imshow("Mask", mask)

# Mostrar transformación de distancia
    cv2.imshow("Dist Transform", dist_transform)

# Mostrar foreground seguro
    cv2.imshow("Sure FG", sure_fg * 255)  # Multiplicar por 255 para visualizar correctamente

# Mostrar fondo seguro
    cv2.imshow("Sure BG", sure_bg)

# Mostrar regiones desconocidas
    cv2.imshow("Unknown", unknown)

    # Dibujar rectángulos alrededor de los objetos detectados
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar contornos pequeños
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result

cv2.waitKey(0)
cv2.destroyAllWindows()

# Ru+ta de la imagen
<<<<<<< HEAD
image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/barco.jpg"
=======
image_path = json.load(open("config.json"))["img_path"] + "barco.jpg"
>>>>>>> 9521a56d5c2466d084a4c3f7a5f6a7895f55e4a7

# Cargar la imagen
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("No se pudo cargar la imagen.")
else:
    # Detectar objetos en el agua
    result = detect_objects(image)
    resized_image = cv2.resize(result, (300, 300))
    
    # Mostrar la imagen con los objetos detectados
    cv2.imshow("Detección de Objetos en el Agua", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


<<<<<<< HEAD















    
   



=======
>>>>>>> 9521a56d5c2466d084a4c3f7a5f6a7895f55e4a7
