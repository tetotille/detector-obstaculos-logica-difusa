import cv2
import numpy as np

def calcular_desenfoque(imagen):
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar la Transformada de Fourier
    dft = cv2.dft(np.float32(gris), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calcular el espectro de magnitud
    magnitud = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    espectro_magnitud = 20 * np.log(magnitud)
    
    # Normalizar el espectro de magnitud
    espectro_magnitud = np.asarray(espectro_magnitud, dtype=np.uint8)
    
    # Calcular la métrica de desenfoque (por ejemplo, el valor promedio)
    promedio_magnitud = np.mean(espectro_magnitud)
    
    return promedio_magnitud

# Cargar la imagen
imagen = cv2.imread('C:/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/sintitulo.jpg')

# Calcular el nivel de desenfoque
nivel_desenfoque = calcular_desenfoque(imagen)
print(f'Nivel de desenfoque: {nivel_desenfoque}')

# Definir un umbral para determinar si está desenfocada
umbral_desenfoque = 150  # Puedes ajustar este valor según tus necesidades
if nivel_desenfoque < umbral_desenfoque:
    print("La imagen está desenfocada")
else:
    print("La imagen está enfocada")
