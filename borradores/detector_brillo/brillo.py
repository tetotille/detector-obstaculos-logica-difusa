import cv2
import numpy as np
from os.path import abspath,dirname,join
from sys import argv
import os

def detectar_y_reducir_brillo(imagen_path, umbral_brillo=200, porcentaje_umbral=50):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    
    # Calcular el histograma
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    
    # Evaluar el porcentaje de píxeles brillantes
    pixeles_brillantes = np.sum(histograma[umbral_brillo:])
    total_pixeles = np.sum(histograma)
    porcentaje_brillo = (pixeles_brillantes / total_pixeles) * 100
    
    print(f'Porcentaje de píxeles brillantes: {porcentaje_brillo}%')
    
    # Verificar si la imagen es demasiado brillante
    if porcentaje_brillo > porcentaje_umbral:
        print("La imagen tiene mucho brillo, se procederá a reducir el brillo.")
        
        # Reducir el brillo
        factor_reduccion = 0.5  # Puedes ajustar este valor
        imagen_reducida = cv2.convertScaleAbs(imagen, alpha=factor_reduccion, beta=0)
        
        # Guardar y mostrar la imagen resultante
        cv2.imwrite('imagen_reducida.jpg', imagen_reducida)
        cv2.imshow('Imagen con brillo reducido', imagen_reducida)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("La imagen no tiene un brillo excesivo.")

# Ru+ta de la imagen
if len(argv)>1:
    filename = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))),"img/sintitulo.jpg")


detectar_y_reducir_brillo(filename)

def reducir_brillo_localmente(imagen, umbral_brillo=200, factor_reduccion=0.5):
    # Convertir la imagen a espacio de color HSV
    hsv_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Separar los canales HSV
    h, s, v = cv2.split(hsv_imagen)
    
    # Identificar píxeles brillantes
    mascara_brillante = v > umbral_brillo
    
    # Reducir el brillo en las áreas brillantes
    v[mascara_brillante] = (v[mascara_brillante] * factor_reduccion).astype(np.uint8)
    
    # Combinar los canales de nuevo
    hsv_imagen_ajustada = cv2.merge([h, s, v])
    
    # Convertir de nuevo a BGR
    imagen_ajustada = cv2.cvtColor(hsv_imagen_ajustada, cv2.COLOR_HSV2BGR)
    
    return imagen_ajustada

def detectar_horizonte(image):
    # Leer la imagen en escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar bordes en la imagen usando Canny edge detector
    edges = cv2.Canny(gray, 170, 400)

    # Definir el kernel para la operación de dilatación horizontal
    horizontal_kernel = np.ones((2, 3), np.uint8)

    # Dilatar los bordes detectados horizontalmente
    dilated_edges_horizontal = cv2.dilate(edges, horizontal_kernel, iterations=3)

    height, width = image.shape[:2]
    min_line_length = int(height * 0.5)  # Longitud mínima relativa 0.5
    max_line_gap = int(height * 0.03)   # Espacio máximo entre líneas relativo 0.03

    # Hough Transform para detectar líneas
    lines = cv2.HoughLinesP(dilated_edges_horizontal, 1, np.pi/90, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        print("No se encontraron líneas.")
        return image, height // 2  # Si no se encuentran líneas, devolver una posición predeterminada del horizonte

    # Encontrar líneas horizontales más largas
    longest_lines = []
    current_line = None
    max_length = 0

    for y in range(dilated_edges_horizontal.shape[0]):
        for x in range(dilated_edges_horizontal.shape[1]):
            if dilated_edges_horizontal[y, x] == 255:
                if current_line is None:
                    current_line = [(x, y)]
                else:
                    current_line.append((x, y))
            elif current_line is not None:
                if len(current_line) > max_length:
                    longest_lines = [current_line]
                    max_length = len(current_line)
                elif len(current_line) == max_length:
                    longest_lines.append(current_line)
                current_line = None

    if not longest_lines:
        print("No se encontraron líneas horizontales largas.")
        return image, height // 2  # Si no se encuentran líneas largas, devolver una posición predeterminada del horizonte

    # Encontrar coordenadas del pixel superior e inferior para cada línea más larga
    top_pixels = []
    bottom_pixels = []

    for line in longest_lines:
        top_pixels.append(line[0])
        bottom_pixels.append(line[-1])

    # Encontrar las coordenadas de la línea que abarca toda la región vertical
    topmost_pixel = min(top_pixels, key=lambda x: x[1])[1]
    bottommost_pixel = max(bottom_pixels, key=lambda x: x[1])[1]

    # Calcular la posición vertical media entre las líneas detectadas
    midpoint_y = (topmost_pixel + bottommost_pixel) // 2

    # Dibujar la línea horizontal en el punto medio
    cv2.line(image, (0, midpoint_y), (image.shape[1] - 1, midpoint_y), (0, 255, 0), 2)

    return image, midpoint_y

# Ruta de la carpeta con las imágenes
ruta_imagenes = 'C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img'
ruta_resultados = 'C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/resultados'

# Asegurarse de que la carpeta de resultados existe
if not os.path.exists(ruta_resultados):
    os.makedirs(ruta_resultados)

# Limitar el número de imágenes para procesar (por ejemplo, 10)
limite_imagenes = 10
contador = 0

for archivo in os.listdir(ruta_imagenes):
    if contador >= limite_imagenes:
        break

    try:
        # Leer la imagen
        ruta_imagen = os.path.join(ruta_imagenes, archivo)
        image = cv2.imread(ruta_imagen)
        if image is None:
            print(f"No se pudo cargar la imagen desde la ruta: {ruta_imagen}")
            continue

        # Reducir el brillo localmente
        imagen_reducida = reducir_brillo_localmente(image)
        
        # Detectar el horizonte en la imagen con brillo reducido
        imagen_con_horizonte, y_horizonte = detectar_horizonte(imagen_reducida)
        
        # Guardar y mostrar la imagen resultante
        resultado_guardado = cv2.imwrite(f'{ruta_resultados}/imagen_con_horizonte_{archivo}', imagen_con_horizonte)
        
        if resultado_guardado:
            print(f"Horizonte detectado en la posición y = {y_horizonte} para la imagen {archivo}")
            resized_image = cv2.resize(imagen_con_horizonte, (300, 300))
            cv2.imshow('Horizontal Line', resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Error al guardar la imagen: {archivo}.")

        contador += 1

    except Exception as e:
        print(f"Error procesando la imagen {archivo}: {e}")
