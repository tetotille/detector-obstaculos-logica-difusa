import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from sys import argv
from utils import crop_horizontal
import time
from horizonte_mar_rojo_cielo_azul import detectar_horizonte
from detector_horizonte import find_horizontal_line
import cv2
from collections import defaultdict

# Paso 1: Cargar la imagen
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/IMG_6830.jpeg")
image2 = cv2.imread(filename)
x, y, a = image2.shape
image_np = cv2.resize(image2, (200, int(x*200/y)))  # Redimensionar para simplificar el procesamiento
height3, width3, canal = image_np.shape
#print(height3, width3)

# Paso 2: Convertir la imagen a un formato de datos
# Reshape la imagen para que cada píxel sea una fila y los valores RGB sean las columnas
pixels = np.reshape(image_np, (-1, 3))

# Normalizar los valores de los píxeles
pixels = pixels / 255.0

print("\npixeles:", pixels)

# Paso 3: Aplicar FCMq
n_clusters = 3  # Número de clusters

start_time = time.perf_counter()
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixels.T, n_clusters, 100, error=0.00005, maxiter=100, init=None)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Tiempo de ejecución hasta el primer resultado: {elapsed_time:.4f} segundos")

# Obtener el índice del cluster más probable para cada píxel
cluster_membership = np.argmax(u, axis=0)

# Paso 4: Reconstruir la imagen segmentada
segmented_image = np.reshape(cluster_membership, (image_np.shape[0], image_np.shape[1])).astype(np.uint8)
#print(segmented_image)
segmented_image_normalized = (segmented_image * (255 / segmented_image.max())).astype(np.uint8)

# Convertir el array de numpy normalizado a un objeto de imagen de Pillow 
#segmented_image_pil = Image.fromarray(segmented_image_normalized)
segmented_image_normalized = np.random.rand(100, 100)  # Ejemplo de array normalizado
segmented_image_normalized = (segmented_image_normalized * 255).astype(np.uint8)

# Paso 5: Calcular la frecuencia de cada cluster
fila_interes, imagen = find_horizontal_line(image2)

#imagen, fila_interes = detectar_horizonte(image_np)
print(fila_interes)

_, image3 =crop_horizontal(segmented_image, fila_interes)
#print(fila_interes)
# Calcular la frecuencia de cada cluster dentro del área de interés
unique, counts = np.unique(image3, return_counts=True)
cluster_frequencies = dict(zip(unique, counts))

# Identificar el cluster con la mayor frecuencia

max_cluster = max(cluster_frequencies, key=cluster_frequencies.get)

# Crear una máscara para resaltar el cluster con mayor frecuencia
mask_max_cluster = (image3 == max_cluster).astype(np.uint8)

# Mostrar la imagen segmentada original
plt.imshow(segmented_image, cmap='viridis')
plt.title('Imagen Segmentada Original')
plt.show()

# Mostrar solo el área de interés debajo de la fila específica
plt.imshow(image3, cmap='viridis')
plt.title('Área de Interés Debajo de la Fila Específica')
plt.show()

# Mostrar la imagen original con el cluster de mayor frecuencia resaltado
plt.imshow(mask_max_cluster)
plt.title(f'Imagen con Cluster {max_cluster} Debajo de la Fila {fila_interes}')
plt.show()


### Parte del contorno
#print(mask_max_cluster)
original = (mask_max_cluster * 255).astype(np.uint8)


# Crea una imagen con transparencia (canal alfa)

#rgba_image = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
mask=cv2.imread("imagen_umbral.png", cv2.IMREAD_GRAYSCALE)

_, mask2=crop_horizontal(mask, fila_interes)

# Leer la imagen original (en color)
original = cv2.cvtColor(original, cv2.IMREAD_COLOR)
height, width, channels = original.shape
height1, width1 = mask2.shape
height2, width2 = mask.shape
print(height, width, channels, height1, width1)

# Verificar las dimensiones de ambas imágenes
if (height, width) != (height1, width1):
    print("Redimensionando la máscara para que coincida con las dimensiones de la imagen original.")
    mask2 = cv2.resize(mask2, (width, height))
else:
    print("Las dimensiones de la máscara ya coinciden con las dimensiones de la imagen original.")


# Asegurarse de que la máscara sea binaria
_, mask_binary = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)

# Crear una imagen de contornos verdes (tamaño de la imagen original)
contour_image = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)

# Establecer los contornos en verde (BGR) usando la máscara binaria
contour_image[mask_binary > 0] = [0, 255, 0]  # Verde en BGR

# Crear una imagen RGBA con el fondo transparente
rgba_image = np.zeros((original.shape[0], original.shape[1], 4), dtype=np.uint8)

# Copiar la imagen original al canal RGB de la imagen RGBA
rgba_image[:, :, :3] = original

# Crear una máscara para el canal alfa (transparencia)
transparency_mask = np.zeros((original.shape[0], original.shape[1]), dtype=np.uint8)
transparency_mask[mask_binary > 0] = 255  # Píxeles de contorno serán completamente opacos

# Copiar la máscara de transparencia al canal alfa de la imagen RGBA
rgba_image[:, :, 3] = transparency_mask

# Aplicar la imagen de contornos al canal alfa de la imagen RGBA
# Para hacer que los contornos sean visibles, combinamos la imagen original con la imagen de contornos
combined_image = cv2.addWeighted(original, 1.0, contour_image, 1.0, 0)

# Crear una imagen RGBA con el fondo transparente y combinar con la imagen de contornos
rgba_image[:, :, :3] = combined_image


#cv2.imwrite(filename, rgba_image)

# Muestra la imagen resultante
cv2.imshow("Imagen con Contornos Superpuestos", rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir la imagen a escala de grises
image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

def segment_and_identify_objects(image_gray, mask_binary, block_size=15, threshold_area=170):
    #lim_maximo_155
# Paso 2: Analizar bloques de 12x12 píxeles
    height, width, chanel = image_gray.shape
    rects = []  # Lista para almacenar los rectángulos detectados
    diagonals = []  # Lista para almacenar las posiciones de rectángulos diagonales
    rect_dict = defaultdict(list)

    for y in range(block_size, height - block_size, block_size):
        for x in range(block_size, width - block_size, block_size):
            # Extraer el bloque de la imagen y de la máscara
            block_image = image_gray[y:y+block_size, x:x+block_size]
            block_mask = mask_binary[y:y+block_size, x:x+block_size]

            # Contar los píxeles negros en el bloque de la imagen en todos los canales
            black_pixel_count = np.sum(np.all(block_image == [0, 0, 0], axis=-1))

            # Verde: canal verde alto y canales rojo y azul bajos
            green_pixels = (block_mask[:, :, 1] > 100) & (block_mask[:, :, 0] < 50) & (block_mask[:, :, 2] < 50)
            green_pixel_count = np.sum(green_pixels)

            # Si el bloque tiene suficientes píxeles negros y un máximo de 40 píxeles verdes
            if black_pixel_count > threshold_area and green_pixel_count <= 40:
                # Inicializar un flag para verificar los bloques adyacentes
                adyacente_verificado = False
                
                # Verificar los 3 bloques adyacentes a la derecha (i, j+1), (i+1, j+1), (i-1, j+1)
                if x + block_size < width and y + block_size < height and y - block_size >= 0:
                    block_mask_right1 = mask_binary[y:y+block_size, x+block_size:x+2*block_size]
                    block_mask_right2 = mask_binary[y+block_size:y+2*block_size, x+block_size:x+2*block_size]
                    block_mask_right3 = mask_binary[y-block_size:y, x+block_size:x+2*block_size]
                    green_pixels_right1 = (block_mask_right1[:, :, 1] > 100) & (block_mask_right1[:, :, 0] < 50) & (block_mask_right1[:, :, 2] < 50)
                    green_pixels_right2 = (block_mask_right2[:, :, 1] > 100) & (block_mask_right2[:, :, 0] < 50) & (block_mask_right2[:, :, 2] < 50)
                    green_pixels_right3 = (block_mask_right3[:, :, 1] > 100) & (block_mask_right3[:, :, 0] < 50) & (block_mask_right3[:, :, 2] < 50)
                    if np.sum(green_pixels_right1) > 15 and np.sum(green_pixels_right2) > 15 and np.sum(green_pixels_right3) > 15:
                        adyacente_verificado = True
                
                # Verificar los 3 bloques adyacentes a la izquierda (i, j-1), (i+1, j-1), (i-1, j-1)
                if x - block_size >= 0 and y + block_size < height and y - block_size >= 0:
                    block_mask_left1 = mask_binary[y:y+block_size, x-block_size:x]
                    block_mask_left2 = mask_binary[y+block_size:y+2*block_size, x-block_size:x]
                    block_mask_left3 = mask_binary[y-block_size:y, x-block_size:x]
                    green_pixels_left1 = (block_mask_left1[:, :, 1] > 100) & (block_mask_left1[:, :, 0] < 50) & (block_mask_left1[:, :, 2] < 50)
                    green_pixels_left2 = (block_mask_left2[:, :, 1] > 100) & (block_mask_left2[:, :, 0] < 50) & (block_mask_left2[:, :, 2] < 50)
                    green_pixels_left3 = (block_mask_left3[:, :, 1] > 100) & (block_mask_left3[:, :, 0] < 50) & (block_mask_left3[:, :, 2] < 50)
                    if np.sum(green_pixels_left1) > 15 and np.sum(green_pixels_left2) > 15 and np.sum(green_pixels_left3) > 15:
                        adyacente_verificado = True
                
                # Verificar los 3 bloques adyacentes hacia abajo (i+1, j), (i+1, j+1), (i+1, j-1)
                if y + block_size < height and x + block_size < width and x - block_size >= 0:
                    block_mask_down1 = mask_binary[y+block_size:y+2*block_size, x:x+block_size]
                    block_mask_down2 = mask_binary[y+block_size:y+2*block_size, x+block_size:x+2*block_size]
                    block_mask_down3 = mask_binary[y+block_size:y+2*block_size, x-block_size:x]
                    green_pixels_down1 = (block_mask_down1[:, :, 1] > 100) & (block_mask_down1[:, :, 0] < 50) & (block_mask_down1[:, :, 2] < 50)
                    green_pixels_down2 = (block_mask_down2[:, :, 1] > 100) & (block_mask_down2[:, :, 0] < 50) & (block_mask_down2[:, :, 2] < 50)
                    green_pixels_down3 = (block_mask_down3[:, :, 1] > 100) & (block_mask_down3[:, :, 0] < 50) & (block_mask_down3[:, :, 2] < 50)
                    if np.sum(green_pixels_down1) > 15 and np.sum(green_pixels_down2) > 15 and np.sum(green_pixels_down3) > 15:
                        adyacente_verificado = True
                
                # Verificar los 3 bloques adyacentes hacia arriba (i-1, j), (i-1, j+1), (i-1, j-1)
                if y - block_size >= 0 and x + block_size < width and x - block_size >= 0:
                    block_mask_up1 = mask_binary[y-block_size:y, x:x+block_size]
                    block_mask_up2 = mask_binary[y-block_size:y, x+block_size:x+2*block_size]
                    block_mask_up3 = mask_binary[y-block_size:y, x-block_size:x]
                    green_pixels_up1 = (block_mask_up1[:, :, 1] > 100) & (block_mask_up1[:, :, 0] < 50) & (block_mask_up1[:, :, 2] < 50)
                    green_pixels_up2 = (block_mask_up2[:, :, 1] > 100) & (block_mask_up2[:, :, 0] < 50) & (block_mask_up2[:, :, 2] < 50)
                    green_pixels_up3 = (block_mask_up3[:, :, 1] > 100) & (block_mask_up3[:, :, 0] < 50) & (block_mask_up3[:, :, 2] < 50)
                    if np.sum(green_pixels_up1) > 15 and np.sum(green_pixels_up2) > 15 and np.sum(green_pixels_up3) > 15:
                        adyacente_verificado = True
                
                    # Si el bloque tiene un adyacente verificado, guardar su posición
                    if adyacente_verificado:
                        rects.append((x, y, block_size, block_size))
                        rect_dict[(x // block_size, y // block_size)].append((x, y, block_size, block_size))
    # Detectar y marcar los rectángulos entre diagonales
    # Paso 2: Verificar y agregar rectángulos intermedios en las diagonales
    for key, rect_list in rect_dict.items():
        for rect1 in rect_list:
            # Considerar solo los vecinos en diagonal (1 bloque en x y 1 bloque en y)
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor_key = (key[0] + dx, key[1] + dy)
                if neighbor_key in rect_dict:
                    for rect2 in rect_dict[neighbor_key]:
                        # Comprobar si los rectángulos están diagonalmente alineados
                        if abs(rect1[0] - rect2[0]) == block_size and abs(rect1[1] - rect2[1]) == block_size:
                            # Encontrar y agregar el rectángulo intermedio
                            x_mid = (rect1[0] + rect2[0]) // 2
                            y_mid = (rect1[1] + rect2[1]) // 2
                            # Asegurarse de que las coordenadas sean válidas y el rectángulo intermedio esté alineado
                            if (x_mid % block_size == 0) and (y_mid % block_size == 0):
                                if (x_mid, y_mid, block_size, block_size) not in rects:
                                    rects.append((x_mid, y_mid, block_size, block_size, "other_color"))

    # Función para marcar rectángulos en rojo
    def mark_rect_red(rect):
        x, y, w, h = rect
        cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Función para marcar rectángulos en azul
    def mark_rect_blue(rect):
        x, y, w, h = rect
        cv2.rectangle(image_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Paso 3: Marcar los rectángulos en la imagen
    for rect in rects:
        if len(rect) == 4:
            mark_rect_red(rect)
        elif len(rect) == 5 and rect[4] == "other_color":
            mark_rect_blue(rect[:4])

    # Mostrar la imagen con los bloques marcados
    cv2.imshow('Segmented Image with Detected Objects', image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Usar la función con la ruta de la imagen y la máscara
segment_and_identify_objects(original, contour_image)