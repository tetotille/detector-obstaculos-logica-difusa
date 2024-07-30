import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from sys import argv
from utils import crop_horizontal
from horizonte_mar_rojo_cielo_azul import detectar_horizonte
import cv2

# Paso 1: Cargar la imagen
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
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

# Paso 3: Aplicar FCM
n_clusters = 3  # Número de clusters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixels.T, n_clusters, 100, error=0.00005, maxiter=100000, init=None)

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

imagen, fila_interes = detectar_horizonte(image_np)

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
mask=cv2.imread("tes.jpeg", cv2.IMREAD_GRAYSCALE)

_, mask2=crop_horizontal(mask, fila_interes)

# Leer la imagen original (en color)
original = cv2.cvtColor(original, cv2.IMREAD_COLOR)
height, width, channels = original.shape
height1, width1 = mask2.shape
height2, width2 = mask.shape
print(height, width, channels,height1, width1, height2, width2 )

# Verificar las dimensiones de ambas imágenes
if (height, width) != (height1, width1):
    print("Redimensionando la máscara para que coincida con las dimensiones de la imagen original.")
    mask2 = cv2.resize(mask, (width, height))
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
def segment_and_identify_objects(image_gray, mask_binary, block_size=15, threshold_area=155):
    #lim_maximo_155
# Paso 2: Analizar bloques de 12x12 píxeles
    height, width, chanel = image_gray.shape
    rects = []  # Lista para almacenar los rectángulos detectados
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
        # Extraer el bloque de la imagen y de la máscara
            block_image = image_gray[y:y+block_size, x:x+block_size]
            block_mask = mask_binary[y:y+block_size, x:x+block_size]

            # Contar los píxeles negros en el bloque de la imagen en todos los canales
            black_pixel_count = np.sum(np.all(block_image == [0, 0, 0], axis=-1))

          # Verde: canal verde alto y canales rojo y azul bajos
            green_pixels = (block_mask[:, :, 1] > 100) & (block_mask[:, :, 0] < 50) & (block_mask[:, :, 2] < 50)
            green_pixel_count = np.sum(green_pixels)
            #print(green_pixel_count)
            #qprint(black_pixel_count)
            # Si el bloque contiene suficientes píxeles negros y tiene contorno en la máscara, marcar el bloque
            if black_pixel_count > threshold_area and green_pixel_count> 70:
                #lim_max_70_green
                rects.append((x, y, block_size, block_size))  # Almacena el 
    n=len(rects)
    print(n)
 # Paso 3: Detectar y marcar rectángulos alineados horizontalmente que cubren toda la fila
    blocks_per_row = int(width // block_size)  # Número de bloques que caben en una fila
    blocks_per_column = int(height // block_size)
    rects_by_column= {}
    rects_by_row = {}
    for rect in rects:
        x, y, w, h = rect
        if x not in rects_by_column:
            rects_by_column[x] = []
        rects_by_column[x].append(rect)
    for rect1 in rects:
        x, y, w, h = rect1
        if y not in rects_by_row:
            rects_by_row[y] = []
        rects_by_row[y].append(rect1)
        # Criterio: número mínimo de rectángulos para marcar la columna en amarillo
# Función para marcar filas en azul
    def mark_row_blue(row_rects):
        for rect in row_rects:
            x, y, w, h = rect
            cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Azul

    # Función para marcar columnas en amarillo
    def mark_column_yellow(column_rects):
        for rect in column_rects:
            x, y, w, h = rect
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Amarillo

    # Función para marcar rectángulos en rojo
    def mark_rect_red(rect):
        x, y, w, h = rect
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rojo

    # Verificar y marcar filas completas
    for y, row_rects in rects_by_row.items():
        row_rects.sort()
        if len(row_rects) == blocks_per_row:
            first_rect_x = row_rects[0][0]
            last_rect_x = row_rects[-1][0] + row_rects[-1][2]
            mark_row_blue(row_rects)

            # Verificar y marcar la fila superior si está alineada
            top_y = y - row_rects[0][3]
            if top_y in rects_by_row:
                top_row_rects = rects_by_row[top_y]
                top_first_x = top_row_rects[0][0]
                top_last_x = top_row_rects[-1][0] + top_row_rects[-1][2]
                if (top_first_x <= first_rect_x <= top_last_x) or (top_first_x <= last_rect_x <= top_last_x):
                    mark_row_blue(top_row_rects)

            # Verificar y marcar la fila inferior si está alineada
            bottom_y = y + row_rects[0][3]
            if bottom_y in rects_by_row:
                bottom_row_rects = rects_by_row[bottom_y]
                bottom_first_x = bottom_row_rects[0][0]
                bottom_last_x = bottom_row_rects[-1][0] + bottom_row_rects[-1][2]
                if (bottom_first_x <= first_rect_x <= bottom_last_x) or (bottom_first_x <= last_rect_x <= bottom_last_x):
                    mark_row_blue(bottom_row_rects)

    # Verificar y marcar columnas completas
    for column_x, column_rects in rects_by_column.items():
        if len(column_rects) == blocks_per_column:
            mark_column_yellow(column_rects)

    # Marcar el resto de los rectángulos en rojo
    for rect in rects:
        x, y, w, h = rect
        color = original[y, x]
        if (color == [0, 0, 255]).all() or (color == [0, 255, 255]).all() or (color == [255, 0, 0]).all():
            continue
        else:
            mark_rect_red(rect)

    # Verificar y marcar columnas completas (incluso si ya están marcadas en azul)
    for column_x, column_rects in rects_by_column.items():
        if len(column_rects) == blocks_per_column:
            for rect in column_rects:
                x, y, w, h = rect
                color = original[y, x]
                if (color == [255, 0, 0]).all():  # Si está marcado en azul
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Amarillo


    # Mostrar la imagen con los bloques marcados
    cv2.imshow('Segmented Image with Detected Objects', image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usar la función con la ruta de la imagen y la máscara
segment_and_identify_objects(original, contour_image)