import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen desde tu archivo
nombre_archivo = "/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/brillo.jpg"
imagen = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)

# Calcular el histograma
histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])

# Encontrar el umbral basado en el histograma (por ejemplo, el valor que corresponde al pico más alto)
umbral = np.argmax(histograma)

# Aplicar una transformación de intensidad para atenuar el brillo
imagen_atenuada = np.where(imagen > umbral, imagen - 50, imagen)

# Mostrar la imagen original y la imagen atenuada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(imagen_atenuada, cmap='gray')
plt.title('Imagen Atenuada')
plt.show()
cv2.imwrite('imagen atenuada.jpg', imagen_atenuada)