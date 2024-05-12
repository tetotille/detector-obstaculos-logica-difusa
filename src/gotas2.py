import cv2
import numpy as np

# Cargar la imagen desde tu archivo
nombre_archivo = '/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/mojado.jpg'
imagen = cv2.imread(nombre_archivo)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización de Otsu para generar la máscara
_, mascara_gotas = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Realizar operaciones morfológicas para mejorar la máscara
kernel = np.ones((5, 5), np.uint8)
mascara_gotas = cv2.morphologyEx(mascara_gotas, cv2.MORPH_CLOSE, kernel)

# Invertir la máscara
mascara_gotas = cv2.bitwise_not(mascara_gotas)

# Encontrar los contornos de las gotas
contornos, _ = cv2.findContours(mascara_gotas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Rellenar los contornos para eliminar las gotas
mascara_gotas = np.zeros_like(mascara_gotas)
cv2.drawContours(mascara_gotas, contornos, -1, (255, 255, 255), thickness=cv2.FILLED)

# Aplicar la máscara a la imagen original
imagen_sin_gotas = cv2.bitwise_and(imagen, imagen, mask=mascara_gotas)

# Mostrar la imagen resultante
cv2.imshow('Imagen', imagen)
cv2.imshow('Imagen sin Gotas', imagen_sin_gotas)
cv2.waitKey(0)
cv2.destroyAllWindows()
