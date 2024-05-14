import numpy as np
import cv2
import matplotlib.pyplot as plt


UNIFORMIDAD = 15

def comparar_color(color_1:np.ndarray, color_2:np.ndarray) -> bool:
    """ Compara la similaridad de los colores respecto a la UNIFORMIDAD definida como constante

    Args:
        color_1 (np.array): Color 1
        color_2 (np.array): Color 2

    Returns:
        bool: retorna true si es similar al color
    """
    return (abs(color_1[0] - color_2[0]) <= UNIFORMIDAD) or (abs(color_1[1] - color_2[1]) <= UNIFORMIDAD) or (abs(color_1[2] - color_2[2]) <= UNIFORMIDAD)

def comparar_uniformidad(pixel_1,pixel_2):
    return abs(pixel_1 - pixel_2) <= UNIFORMIDAD

# Carga la imagen
ruta = "/home/tille/Desktop/Tesis/code/img/atardecer.jpg"
imagen = cv2.imread(ruta)

resized_image = cv2.resize(imagen, (300, 300))

# Convert to HSV color space
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Apply Gaussian blur to each channel separately
blurred_h = cv2.GaussianBlur(hsv_image[:, :, 0], (5, 5), 0)
blurred_s = cv2.GaussianBlur(hsv_image[:, :, 1], (5, 5), 0)
blurred_v = cv2.GaussianBlur(hsv_image[:, :, 2], (5, 5), 0)

# Combine blurred channels back to HSV
blurred_hsv = np.dstack((blurred_h, blurred_s, blurred_v))

# Convert back to BGR color space
blurred_image = cv2.cvtColor(blurred_hsv, cv2.COLOR_HSV2BGR)

# Convierte la imagen a un arreglo de NumPy
imagen_numpy = np.array(blurred_image)


# Define el rango de uniformidad de color

# Crea una matriz vacía para almacenar las asociaciones de píxeles
asociaciones_pixeles = np.zeros(imagen_numpy.shape, dtype=np.int32)

resultado = np.zeros((imagen_numpy.shape[0],imagen_numpy.shape[1]))

# Recorre cada píxel de la imagen

for fila in range(imagen_numpy.shape[0]): # O[x^4]
    for columna in range(imagen_numpy.shape[1]):
        puntaje = 0
        # Obtiene el color del píxel actual
        color_actual = imagen_numpy[fila, columna]
        print(f"fila: {fila} columna: {columna}",end='\r')
        try:
            if comparar_color(color_actual, imagen_numpy[fila-1, columna-1]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila-1,columna]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila-1,columna+1]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila,columna-1]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila,columna+1]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila+1,columna-1]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila+1,columna]):
                puntaje += 1
        except:
            pass
        try:
            if comparar_color(color_actual, imagen_numpy[fila+1,columna+1]):
                puntaje += 1
        except:
            pass
            
        resultado[fila,columna] = int(puntaje * 255 / 8)
        

# Visualiza las asociaciones de píxeles

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.imshow(resultado)
ax2.imshow(blurred_image)
plt.show()
print("HOLA")
cv2.imwrite("imagen_modificada.jpg", resultado)