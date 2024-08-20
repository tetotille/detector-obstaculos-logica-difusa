import skfuzzy as fuzz
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, abspath, join
from sys import argv
from utils import crop_horizontal
from horizonte_mar_rojo_cielo_azul import detectar_horizonte

def filter_h(img):
     #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h,_,_ = cv2.split(img_hsv)
    
    h = 255 - h

    num_bins = 256
    fuzzy_hist = np.zeros(num_bins)

    x_intensities = np.arange(0, 256, 1)
    
    for i in range(num_bins):
        # Crear la función de pertenencia trapezoidal para el intervalo
        if i == 0:
            # Trapezoide inicial
            mf = fuzz.trapmf(x_intensities, [0, 0, 1, 2])

        elif i == num_bins - 1:
            # Trapezoide final
            mf = fuzz.trapmf(x_intensities, [254, 255, 255, 255])

        else:
            # Trapezoide intermedio
            mf = fuzz.trapmf(x_intensities, [i-1, i, i+1, i+2])
        
        # Calcular el grado de pertenencia de cada píxel a este intervalo
        membership_values = fuzz.interp_membership(x_intensities, mf, h)
        
        # Sumar los valores de pertenencia para formar el histograma difuso
        fuzzy_hist[i] = np.sum(membership_values)
    
    max_index = np.argmax(fuzzy_hist)
    most_frequent_intensity = x_intensities[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    h_filtrada = np.copy(h)
    h_filtrada[(h >= most_frequent_intensity-10) & (h <= most_frequent_intensity+10)] = 0
    return h_filtrada

def filter_s(img_path):
    #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)
    
    _,s,_ = cv2.split(img_hsv)
    
    s = 255 - s

    hist, bins = np.histogram(s.ravel(), 256, [0, 256])

    max_index = np.argmax(hist)
    most_frequent_intensity = bins[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    s_filtrada = np.copy(s)
    s_filtrada[(s >= most_frequent_intensity-10) & (s <= most_frequent_intensity+10)] = 0

    return s_filtrada

def segment_and_identify_objects(image_gray, mask_binary, block_size=15, threshold_area=155):
        #lim_maximo_155
    # Paso 2: Analizar bloques de 12x12 píxeles
        height, width, chanel = image_gray.shape
        rects = []  # Lista para almacenar los rectángulos detectados
        for y in range(block_size, height - block_size, block_size):
            for x in range(block_size, width - block_size, block_size):
                # Extraer el bloque de la imagen y de la máscara
                block_image = image_gray[y:y+block_size, x:x+block_size]
                block_mask = mask_binary[y:y+block_size, x:x+block_size]

                # Contar los píxeles negros en el bloque de la imagen en todos los canales
                black_pixel_count = np.sum(np.all(block_image == [0, 0, 0], axis=-1))

                # Verde: canal verde alto y canales rojo y azul bajos
                green_pixels = (block_mask[:, :, 1] > 100) & (block_mask[:, :, 0] < 50) & (block_mask[:, :, 2] < 50)
                green_pixel_count = np.sum(green_pixels)

                # Si el bloque tiene suficientes píxeles negros y un máximo de 70 píxeles verdes
                if black_pixel_count > threshold_area and green_pixel_count <= 40:
                    # Inicializar un flag para verificar los bloques adyacentes
                    adyacente_verificado = False
                    
                    # Verificar los 3 bloques adyacentes a la derecha (i, j+1), (i+1, j+1), (i-1, j+1)
                    if x + block_size < width and y + block_size < height and y - block_size >= 0:
                        block_mask_right1 = mask_binary[y:y+block_size, x+block_size:x+2*block_size]
                        block_mask_right2 = mask_binary[y+block_size:y+2*block_size, x+block_size:x+2*block_size]
                        block_mask_right3 = mask_binary[y-block_size:y, x+block_size:x+2*block_size]
                        green_pixels_right1 = (block_mask_right1[:, :, 1] > 100) & (block_mask_right1[:, :, 0] < 50) & (block_mask_right1[:, :, 2] < 50)
                        green_pixels_right2 = (block_mask_right2[:, :, 1] > 100) & (block_mask_right2[:, :, 0] < 50) & (block_mask_right2[:, :, 2] < 50)
                        green_pixels_right3 = (block_mask_right3[:, :, 1] > 100) & (block_mask_right3[:, :, 0] < 50) & (block_mask_right3[:, :, 2] < 50)
                        if np.sum(green_pixels_right1) > 15 and np.sum(green_pixels_right2) > 15 and np.sum(green_pixels_right3) > 15:
                            adyacente_verificado = True
                    
                    # Verificar los 3 bloques adyacentes a la izquierda (i, j-1), (i+1, j-1), (i-1, j-1)
                    if x - block_size >= 0 and y + block_size < height and y - block_size >= 0:
                        block_mask_left1 = mask_binary[y:y+block_size, x-block_size:x]
                        block_mask_left2 = mask_binary[y+block_size:y+2*block_size, x-block_size:x]
                        block_mask_left3 = mask_binary[y-block_size:y, x-block_size:x]
                        green_pixels_left1 = (block_mask_left1[:, :, 1] > 100) & (block_mask_left1[:, :, 0] < 50) & (block_mask_left1[:, :, 2] < 50)
                        green_pixels_left2 = (block_mask_left2[:, :, 1] > 100) & (block_mask_left2[:, :, 0] < 50) & (block_mask_left2[:, :, 2] < 50)
                        green_pixels_left3 = (block_mask_left3[:, :, 1] > 100) & (block_mask_left3[:, :, 0] < 50) & (block_mask_left3[:, :, 2] < 50)
                        if np.sum(green_pixels_left1) > 15 and np.sum(green_pixels_left2) > 15 and np.sum(green_pixels_left3) > 15:
                            adyacente_verificado = True
                    
                    # Verificar los 3 bloques adyacentes hacia abajo (i+1, j), (i+1, j+1), (i+1, j-1)
                    if y + block_size < height and x + block_size < width and x - block_size >= 0:
                        block_mask_down1 = mask_binary[y+block_size:y+2*block_size, x:x+block_size]
                        block_mask_down2 = mask_binary[y+block_size:y+2*block_size, x+block_size:x+2*block_size]
                        block_mask_down3 = mask_binary[y+block_size:y+2*block_size, x-block_size:x]
                        green_pixels_down1 = (block_mask_down1[:, :, 1] > 100) & (block_mask_down1[:, :, 0] < 50) & (block_mask_down1[:, :, 2] < 50)
                        green_pixels_down2 = (block_mask_down2[:, :, 1] > 100) & (block_mask_down2[:, :, 0] < 50) & (block_mask_down2[:, :, 2] < 50)
                        green_pixels_down3 = (block_mask_down3[:, :, 1] > 100) & (block_mask_down3[:, :, 0] < 50) & (block_mask_down3[:, :, 2] < 50)
                        if np.sum(green_pixels_down1) > 15 and np.sum(green_pixels_down2) > 15 and np.sum(green_pixels_down3) > 15:
                            adyacente_verificado = True
                    
                    # Verificar los 3 bloques adyacentes hacia arriba (i-1, j), (i-1, j+1), (i-1, j-1)
                    if y - block_size >= 0 and x + block_size < width and x - block_size >= 0:
                        block_mask_up1 = mask_binary[y-block_size:y, x:x+block_size]
                        block_mask_up2 = mask_binary[y-block_size:y, x+block_size:x+2*block_size]
                        block_mask_up3 = mask_binary[y-block_size:y, x-block_size:x]
                        green_pixels_up1 = (block_mask_up1[:, :, 1] > 100) & (block_mask_up1[:, :, 0] < 50) & (block_mask_up1[:, :, 2] < 50)
                        green_pixels_up2 = (block_mask_up2[:, :, 1] > 100) & (block_mask_up2[:, :, 0] < 50) & (block_mask_up2[:, :, 2] < 50)
                        green_pixels_up3 = (block_mask_up3[:, :, 1] > 100) & (block_mask_up3[:, :, 0] < 50) & (block_mask_up3[:, :, 2] < 50)
                        if np.sum(green_pixels_up1) > 15 and np.sum(green_pixels_up2) > 15 and np.sum(green_pixels_up3) > 15:
                            adyacente_verificado = True
                    
                    # Si cualquiera de las direcciones tiene 3 bloques adyacentes con suficientes píxeles verdes
                    if adyacente_verificado:
                        rects.append((x, y, block_size, block_size))
        n=len(rects)
        print(n)
        # Función para marcar rectángulos en rojo
        def mark_rect_red(rect):
            x, y, w, h = rect
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rojo

        # Verificar y marcar filas completas
        # Marcar el resto de los rectángulos en rojo
        for rect in rects:
            x, y, w, h = rect
            color = original[y, x]
            if (color == [0, 0, 255]).all() or (color == [0, 255, 255]).all() or (color == [255, 0, 0]).all():
                continue
            else:
                mark_rect_red(rect)
        cv2.imshow('Segmented Image with Detected Objects', image_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":

    # Ruta a la imagen
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
    img=cv2.imread(filename)
    
    cv2.imshow('imagen final.png',filter_h(img))
    cv2.waitKey(0)
    x,y,a = img.shape
    imag=cv2.resize(img, (200, int(x*200/y)))
    ima, fila_interes = detectar_horizonte(imag)
    imagen=cv2.resize(img, (200, int(x*200/y)))
    _, image3 =crop_horizontal(filter_h(imagen), fila_interes)
    print(fila_interes)

    # Crea una imagen con transparencia (canal alfa)
    mask=cv2.imread("imagen_umbral.png", cv2.IMREAD_GRAYSCALE)
    x,y = mask.shape
    _, mask2=crop_horizontal(mask, fila_interes)
    # Leer la imagen original (en color)
   
    original = cv2.cvtColor(image3, cv2.IMREAD_COLOR)
    
    # Crear una máscara para los píxeles negros
    mask = cv2.inRange(original, np.array([0, 0, 0]), np.array([15, 15, 15]))  # Ajusta el rango según sea necesario

    # Crear imágenes en negro y blanco
    white_image = np.full_like(original, 255)
    black_image = np.zeros_like(original)

    # Aplicar la máscara para obtener la imagen con negros convertidos a blancos y el resto a negro
    original = cv2.bitwise_and(white_image, white_image, mask=mask) + cv2.bitwise_and(black_image, black_image, mask=cv2.bitwise_not(mask))


    x,y,a = original.shape
    height, width, channels = original.shape
    height1, width1 = mask2.shape
    print(height, width, channels,height1, width1 )
    # Verificar las dimensiones de ambas imágenes

    if (height, width) != (height1, width1):
        print("Redimensionando la máscara para que coincida con las dimensiones de la imagen original.")
        mask2 = cv2.resize(mask2, (width, height))
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
# Usar la función con la ruta de la imagen y la máscara
    segment_and_identify_objects(original, contour_image)