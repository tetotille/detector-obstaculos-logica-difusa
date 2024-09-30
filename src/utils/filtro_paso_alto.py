import cv2
import numpy as np

# Cargar la imagen desde tu archivo
nombre_archivo = '/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/barco.jpg'
imagen = cv2.imread(nombre_archivo)

# Cambia el tamaño de la imagen a 640x480
imagen_resized =cv2.resize(imagen,(640, 480))

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen_resized, cv2.COLOR_BGR2GRAY)

# Aplicar el filtro de paso alto (en este caso, utilizando el filtro Laplaciano)
filtro_paso_alto = cv2.Laplacian(imagen_gris, cv2.CV_64F)

# Convertir la imagen resultante a valores absolutos
filtro_paso_alto = np.uint8(np.absolute(filtro_paso_alto))

# Atenuar el brillo de la imagen original restando el resultado del filtro de paso alto
imagen_atenuada = cv2.subtract(imagen_gris, filtro_paso_alto)

# Mostrar la imagen original, el filtro de paso alto y la imagen atenuada
cv2.imwrite('Filtro de Paso Alto.jpg', filtro_paso_alto)
cv2.imshow('Imagen Original', imagen_resized)
cv2.imshow('Filtro de Paso Alto', filtro_paso_alto)
cv2.imshow('Imagen Atenuada', imagen_atenuada)
cv2.waitKey(0)
cv2.destroyAllWindows()