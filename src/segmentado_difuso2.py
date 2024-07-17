import numpy as np
from PIL import Image
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skimage import measure
from os.path import dirname, abspath, join
from sys import argv

# Paso 1: Cargar la imagen
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
image2 = Image.open(filename)
x, y = image2.size
image = image2.resize((200, int(x*200/y)))  # Redimensionar para simplificar el procesamiento
image_np = np.array(image)


# Paso 2: Convertir la imagen a un formato de datos
# Reshape la imagen para que cada píxel sea una fila y los valores RGB sean las columnas
pixels = np.reshape(image_np, (-1, 3))

# Normalizar los valores de los píxeles
pixels = pixels / 255.0

# Paso 3: Aplicar FCM
n_clusters = 4  # Número de clusters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    pixels.T, n_clusters, 100, error=0.00005, maxiter=100000, init=None)

# Obtener el índice del cluster más probable para cada píxel
cluster_membership = np.argmax(u, axis=0)

# Paso 4: Reconstruir la imagen segmentada
segmented_image = np.reshape(cluster_membership, (image_np.shape[0], image_np.shape[1]))

# Paso 5: Identificar regiones conectadas para cada cluster
regions = measure.label(segmented_image, connectivity=2)

# Calcular el tamaño de cada región
region_props = measure.regionprops(regions)
region_areas = [region.area for region in region_props]

# Identificar la región más grande
largest_region_idx = np.argmax(region_areas)
largest_region_label = region_props[largest_region_idx].label

# Crear una máscara para la región más grande
mask = (regions == largest_region_label).astype(np.uint8)

# Aplicar la máscara al objeto original
extracted_object = image_np * mask[:,:,np.newaxis]

# Crear una máscara donde los valores del objeto detectado sean negros y el resto sea blanco
object_value = 0  # Suponiendo que el valor del objeto es 1 después de la segmentación
mask = np.where(extracted_object == object_value, 0, 255).astype(np.uint8)

# Convertir el array de numpy a un objeto de imagen de Pillow
mask_image_pil = Image.fromarray(mask)
# Paso 5: Comparar la imagen original con el objeto extraído
difference_image = np.abs(image_np - extracted_object)

# Mostrar la imagen original, el objeto extraído y la diferencia
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Imagen Original')
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.title('Objeto Extraído')
plt.imshow(extracted_object)


plt.subplot(1, 3, 3)
plt.title('Diferencia')
plt.imshow(mask_image_pil)

plt.show()