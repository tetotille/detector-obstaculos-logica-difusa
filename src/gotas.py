import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Cargar la imagen desde tu archivo
nombre_archivo = '/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/mojado.jpg'
imagen = cv2.imread(nombre_archivo)

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Calcular la matriz de co-ocurrencia de niveles de gris (GLCM)
distancia = 1
angulo = 0
glcm = graycomatrix(imagen_gris, [distancia], [angulo], levels=256, symmetric=True, normed=True)

# Calcular las propiedades de la textura de Haralick
propiedades = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
atributos = [graycoprops(glcm, prop) for prop in propiedades]

# Concatenar los atributos en un solo vector
atributos_concatenados = np.hstack(atributos)

# Definir un umbral para identificar las gotas de agua
umbral = 0.1

# Aplicar umbralización para identificar las regiones de la imagen que contienen gotas
mascara_gotas = cv2.compare(atributos_concatenados, umbral, cv2.CMP_GT)

# Invertir la máscara de las gotas
mascara_gotas = cv2.bitwise_not(mascara_gotas)

# Aplicar la máscara a la imagen original para eliminar las gotas

# Convertir la máscara a CV_8U
mascara_gotas = mascara_gotas.astype(np.uint8)
mascara_gotas[mascara_gotas > 0] = 255

# Asegurarse de que la máscara tenga la misma forma que la imagen original
mascara_gotas = cv2.resize(mascara_gotas, (imagen.shape[1], imagen.shape[0]))

# Aplicar la máscara a la imagen original
imagen_sin_gotas = cv2.bitwise_and(imagen, imagen, mask=mascara_gotas)

# Mostrar la imagen resultante
cv2.imshow('Imagen sin Gotas', imagen_sin_gotas)
cv2.waitKey(0)
cv2.destroyAllWindows()
