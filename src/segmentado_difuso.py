import numpy as np
from PIL import Image
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from sys import argv

if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/IMG_6830.jpeg")
image2 = Image.open(filename)
x, y = image2.size
image = image2.resize((200, int(x*200/y)))  # Redimensionar para simplificar el procesamiento
image_np = np.array(image)

# Paso 2: Convertir la imagen a un formato de datos
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

# Mostrar la imagen original y la imagen segmentada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title('Imagen Segmentada')
plt.imshow(segmented_image, cmap='viridis')

plt.show()
