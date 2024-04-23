import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_orig = cv2.imread('/home/tille/Desktop/Tesis/code/img/barco.jpg')

image = image_orig[1010:,:]

# cv2.imshow('RGB Image',image )
# cv2.waitKey(0)
# Separar los canales de color
b, g, r = cv2.split(image)

# Calcular los histogramas de cada canal de color
hist_b, bins_b = np.histogram(b.flatten(), 256, [0, 256])
hist_g, bins_g = np.histogram(g.flatten(), 256, [0, 256])
hist_r, bins_r = np.histogram(r.flatten(), 256, [0, 256])

rojo = [75,126]
verde = [78,140]
azul = [87,150]

plt.subplot(131)  # Fila 1, Columna 1
plt.bar(bins_b[:-1], hist_b, width=1, label='Azul')
plt.xlabel('Umbrales')
plt.ylabel('Frecuencia')
plt.xlim([0, 256])
plt.title('Histograma Azul')

plt.subplot(132)  # Fila 1, Columna 2
plt.bar(bins_g[:-1], hist_g, width=1, label='Verde')
plt.xlabel('Umbrales')
plt.ylabel('Frecuencia')
plt.xlim([0, 256])
plt.title('Histograma Verde')

plt.subplot(133)  # Fila 1, Columna 3
plt.bar(bins_r[:-1], hist_r, width=1, label='Rojo')
plt.xlabel('Umbrales')
plt.ylabel('Frecuencia')
plt.xlim([0, 256])
plt.title('Histograma Rojo')

plt.tight_layout()
plt.show()

import cv2
import numpy as np

# Cargar la imagen
image = image_orig


# Convertir la imagen a HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir los rangos de colores para azul (agua) en HSV
lower_blue = np.array([95, 100, 50])
upper_blue = np.array([130, 255, 255])

# Crear máscara para el rango de color azul
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Aplicar suavizado gaussiano a la máscara
mask_blue = cv2.GaussianBlur(mask_blue, (5, 5), 0)

# Aplicar erosión y dilatación a la máscara
kernel = np.ones((3, 3), np.uint8)
mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(image, image, mask=mask_blue)

# Guardar la imagen resultante
cv2.imwrite('output_image.jpg', result)

exit()

# Definir los rangos de colores para rojo, verde y azul en BGR
rojo = [140, 156]
verde = [170, 178]
azul = [185, 190]

lower_red = np.array([0, 0, rojo[0]])
upper_red = np.array([255, 255,rojo[1]])

lower_green = np.array([0, verde[0], 0])
upper_green = np.array([255, verde[1], 255])

lower_blue = np.array([azul[0], 0, 0])
upper_blue = np.array([azul[1], 255,255 ])

# Crear máscaras para cada rango de color
mask_red = cv2.inRange(image, lower_red, upper_red)
mask_green = cv2.inRange(image, lower_green, upper_green)
mask_blue = cv2.inRange(image, lower_blue, upper_blue)

# Combinar las máscaras
combined_mask = cv2.bitwise_or(mask_red, mask_green)
combined_mask = cv2.bitwise_or(combined_mask, mask_blue)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(image, image, mask=combined_mask)

# Convertir los píxeles fuera de los rangos definidos a gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.expand_dims(gray, axis=2)  # Expandir dimensiones del array grayscale

result[combined_mask == 0] = 0
# Guardar la imagen resultante
cv2.imwrite('output_image.jpg', result)
