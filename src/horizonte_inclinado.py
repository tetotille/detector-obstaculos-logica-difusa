import cv2
import numpy as np

def encontrar_linea_mas_larga(contornos):
    # Inicializar variables para almacenar la línea más larga encontrada
    max_length = 0
    best_line = None

    # Iterar sobre todos los contornos
    for contorno in contornos:
        # Calcular la longitud del contorno
        longitud = cv2.arcLength(contorno, True)
        # Actualizar la línea más larga si encontramos una más larga
        if longitud > max_length:
            max_length = longitud
            # Ajustar la línea a la del contorno
            best_line = cv2.approxPolyDP(contorno, 0.02 * longitud, True)

    return best_line

# Función para detectar el horizonte, tanto horizontal como inclinado
def detectar_horizonte(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta del archivo y asegúrate de que el archivo exista.")

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de Canny para detectar los bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Encontrar los contornos en los bordes detectados
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar la línea más larga entre los contornos
    linea_mas_larga = encontrar_linea_mas_larga(contornos)

    # Dibujar la línea más larga encontrada
    if linea_mas_larga is not None:
        cv2.drawContours(image, [linea_mas_larga], -1, (0, 255, 0), 2)

    # Mostrar la imagen con el horizonte detectado
    resized_image = cv2.resize(image, (600, 600))  # Aumentar el tamaño para mejor visualización
    cv2.imshow('Horizon Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta a la imagen
image_path = "C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/sintitulo.jpg"
detectar_horizonte(image_path)

