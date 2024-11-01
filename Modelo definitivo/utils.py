import cupy as cp
import cv2
from cupyx.scipy.ndimage import convolve
def crop_horizontal(imagen, indice_vertical):
    """
    Recorta una imagen a color horizontalmente en un índice dado usando CuPy.
    Args:
        imagen: Una imagen a color en formato CuPy.
        indice_vertical: El índice vertical donde se realizará el recorte.
    Returns:
        Una tupla que contiene dos imágenes: la parte superior y la parte inferior.
    """

    # Verifica si indice_vertical es un array de más de un elemento
    if isinstance(indice_vertical, cp.ndarray):
        if indice_vertical.size > 1:
            raise ValueError("indice_vertical debe ser un valor único, pero se recibió un array de más de un elemento.")
        indice_vertical = indice_vertical.item()

    if indice_vertical < 0 or indice_vertical >= imagen.shape[0]:
        raise ValueError("El índice vertical está fuera de los límites de la imagen.")
    
    parte_superior = imagen[:indice_vertical, :]
    parte_inferior = imagen[indice_vertical:, :]
    
    return parte_superior, parte_inferior

def segment_and_identify_objects(image_gray, mask_binary, original, block_size=15, threshold_area=60):
    # Paso 2: Analizar bloques de 15x15 píxeles
    height, width = image_gray.shape[:2]
    # Generar índices de los bloques
    y_indices = cp.arange(block_size, height - block_size, block_size)
    x_indices = cp.arange(block_size, width - block_size, block_size)
    
    # Crear grids de coordenadas
    y_grid, x_grid = cp.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Extraer los bloques de la imagen y la máscara utilizando broadcast
    block_images = cp.array([image_gray[y:y+block_size, x:x+block_size] for y, x in zip(y_grid.ravel(), x_grid.ravel())])
    block_masks = cp.array([mask_binary[y:y+block_size, x:x+block_size] for y, x in zip(y_grid.ravel(), x_grid.ravel())])
    
    # Contar los píxeles negros en los bloques
    black_pixel_counts = cp.sum(cp.all(block_images == 0, axis=-1), axis=(1, 2))
    
    # Identificar píxeles verdes en los bloques
    green_pixels = (block_masks[:, :, :, 1] > 100) & (block_masks[:, :, :, 0] < 50) & (block_masks[:, :, :, 2] < 50)
    green_pixel_counts = cp.sum(green_pixels, axis=(1, 2))
    #print(green_pixel_counts)
    # Crear un vector booleano que indique qué bloques cumplen la condición inicial
    valid_blocks = (black_pixel_counts > threshold_area) & (green_pixel_counts <= 30)
    #print(black_pixel_counts)
    # Verificar bloques adyacentes
    rects = []
    for idx, (y, x) in enumerate(zip(y_grid.ravel(), x_grid.ravel())):
        if not valid_blocks[idx]:
            continue

        # Verificar bloques adyacentes
        adyacente_verificado = False

        # Verificar los 3 bloques adyacentes a la derecha (i, j+1), (i+1, j+1), (i-1, j+1)
        if x + block_size < width and y + block_size < height and y - block_size >= 0:
            block_mask_right1 = mask_binary[y:y+block_size, x+block_size:x+2*block_size]
            block_mask_right2 = mask_binary[y+block_size:y+2*block_size, x+block_size:x+2*block_size]
            block_mask_right3 = mask_binary[y-block_size:y, x+block_size:x+2*block_size]
            green_pixels_right1 = (block_mask_right1[:, :, 1] > 100) & (block_mask_right1[:, :, 0] < 50) & (block_mask_right1[:, :, 2] < 50)
            green_pixels_right2 = (block_mask_right2[:, :, 1] > 100) & (block_mask_right2[:, :, 0] < 50) & (block_mask_right2[:, :, 2] < 50)
            green_pixels_right3 = (block_mask_right3[:, :, 1] > 100) & (block_mask_right3[:, :, 0] < 50) & (block_mask_right3[:, :, 2] < 50)
            if cp.sum(green_pixels_right1) > 20 and cp.sum(green_pixels_right2) > 20 and cp.sum(green_pixels_right3) > 20:
                adyacente_verificado = True
        
        # Verificar los 3 bloques adyacentes a la izquierda (i, j-1), (i+1, j-1), (i-1, j-1)
        if x - block_size >= 0 and y + block_size < height and y - block_size >= 0:
            block_mask_left1 = mask_binary[y:y+block_size, x-block_size:x]
            block_mask_left2 = mask_binary[y+block_size:y+2*block_size, x-block_size:x]
            block_mask_left3 = mask_binary[y-block_size:y, x-block_size:x]
            green_pixels_left1 = (block_mask_left1[:, :, 1] > 100) & (block_mask_left1[:, :, 0] < 50) & (block_mask_left1[:, :, 2] < 50)
            green_pixels_left2 = (block_mask_left2[:, :, 1] > 100) & (block_mask_left2[:, :, 0] < 50) & (block_mask_left2[:, :, 2] < 50)
            green_pixels_left3 = (block_mask_left3[:, :, 1] > 100) & (block_mask_left3[:, :, 0] < 50) & (block_mask_left3[:, :, 2] < 50)
            if cp.sum(green_pixels_left1) > 20 and cp.sum(green_pixels_left2) > 20 and cp.sum(green_pixels_left3) > 20:
                adyacente_verificado = True
        
        # Verificar los 3 bloques adyacentes hacia abajo (i+1, j), (i+1, j+1), (i+1, j-1)
        if y + block_size < height and x + block_size < width and x - block_size >= 0:
            block_mask_down1 = mask_binary[y+block_size:y+2*block_size, x:x+block_size]
            block_mask_down2 = mask_binary[y+block_size:y+2*block_size, x+block_size:x+2*block_size]
            block_mask_down3 = mask_binary[y+block_size:y+2*block_size, x-block_size:x]
            green_pixels_down1 = (block_mask_down1[:, :, 1] > 100) & (block_mask_down1[:, :, 0] < 50) & (block_mask_down1[:, :, 2] < 50)
            green_pixels_down2 = (block_mask_down2[:, :, 1] > 100) & (block_mask_down2[:, :, 0] < 50) & (block_mask_down2[:, :, 2] < 50)
            green_pixels_down3 = (block_mask_down3[:, :, 1] > 100) & (block_mask_down3[:, :, 0] < 50) & (block_mask_down3[:, :, 2] < 50)
            if cp.sum(green_pixels_down1) > 20 and cp.sum(green_pixels_down2) > 20 and cp.sum(green_pixels_down3) > 20:
                adyacente_verificado = True
        
        # Verificar los 3 bloques adyacentes hacia arriba (i-1, j), (i-1, j+1), (i-1, j-1)
        if y - block_size >= 0 and x + block_size < width and x - block_size >= 0:
            block_mask_up1 = mask_binary[y-block_size:y, x:x+block_size]
            block_mask_up2 = mask_binary[y-block_size:y, x+block_size:x+2*block_size]
            block_mask_up3 = mask_binary[y-block_size:y, x-block_size:x]
            green_pixels_up1 = (block_mask_up1[:, :, 1] > 100) & (block_mask_up1[:, :, 0] < 50) & (block_mask_up1[:, :, 2] < 50)
            green_pixels_up2 = (block_mask_up2[:, :, 1] > 100) & (block_mask_up2[:, :, 0] < 50) & (block_mask_up2[:, :, 2] < 50)
            green_pixels_up3 = (block_mask_up3[:, :, 1] > 100) & (block_mask_up3[:, :, 0] < 50) & (block_mask_up3[:, :, 2] < 50)
            if cp.sum(green_pixels_up1) > 20 and cp.sum(green_pixels_up2) > 20 and cp.sum(green_pixels_up3) > 20:
                adyacente_verificado = True
            # Función para marcar rectángulos en rojo
        def mark_rect_red(rect):
            x, y, w, h = rect
            original[y:y+h, x:x+w, :] = cp.array([0, 0, 255])  # Marcar en rojo
        # Si cualquiera de las direcciones tiene 3 bloques adyacentes con suficientes píxeles verdes
        if adyacente_verificado:
            rects.append((x, y, block_size, block_size))  # Agregar rectángulo
            mark_rect_red((x, y, block_size, block_size))  # Llamar a la función para marcar en rojo
            
    n = len(rects)
    #print(n)
    


    
    # Convertir a un formato que pueda mostrar la imagen
    """image_to_show = cp.asnumpy(original)

    # Mostrar la imagen con los bloques marcados
    cv2.imshow('Segmented Image with Detected Objects', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    return rects, block_size

# Usar la función con la ruta de la imagen, la máscara y la imagen original en forma de array de CuPy

def resize_image_bgr(image, new_shape):
    orig_height, orig_width, channels = image.shape
    new_height, new_width = new_shape

    scale_y = orig_height / new_height
    scale_x = orig_width / new_width

    y = cp.arange(new_height) * scale_y
    x = cp.arange(new_width) * scale_x
    x_grid, y_grid = cp.meshgrid(x, y)

    x0 = cp.floor(x_grid).astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, orig_width - 1)
    y0 = cp.floor(y_grid).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, orig_height - 1)

    x_weight = x_grid - x0
    y_weight = y_grid - y0

    resized_image = cp.zeros((new_height, new_width, channels), dtype=image.dtype)
    for c in range(channels):
        Ia = image[y0, x0, c]
        Ib = image[y1, x0, c]
        Ic = image[y0, x1, c]
        Id = image[y1, x1, c]

        resized_image[:, :, c] = (
            Ia * (1 - x_weight) * (1 - y_weight) +
            Ib * (1 - x_weight) * y_weight +
            Ic * x_weight * (1 - y_weight) +
            Id * x_weight * y_weight
        )

    resized_image = cp.clip(resized_image, 0, 255)
    return resized_image.astype(cp.uint8)
# Usar la función con la ruta de la imagen, la máscara y la imagen original en forma de array de CuPy

def hacer_mascara2(image3, fila_interes):
    # Leer la máscara (suponiendo que ya está en la GPU como array de CuPy)
    mask = cp.asarray(cv2.imread("imagen_umbral.png", cv2.IMREAD_GRAYSCALE))

    # Obtener las dimensiones de la máscara
    x, y = mask.shape

    # Asumiendo que `crop_horizontal` también está en CuPy y devuelve un array de CuPy
    _, mask2 = crop_horizontal(mask, fila_interes)

    image3_cupy = cp.asarray(image3)

    # Verifica si la imagen tiene 3 canales
    if image3_cupy.ndim == 3:
        original = image3_cupy[:, :, ::-1]  # Convertir de BGR a RGB
    else:
        original = image3_cupy  # No es necesario cambiar si es escala de grises

    # Crear la máscara basada en el rango de valores
    if original.ndim == 3:
        mask = cp.logical_and(
            cp.all(original >= 0, axis=-1),
            cp.all(original <= 15, axis=-1)
        ).astype(cp.uint8) * 255
    else:
        mask = cp.logical_and(
            original >= 0,
            original <= 15
        ).astype(cp.uint8) * 255

    # Crear imágenes blancas y negras
    white_image = cp.full(original.shape, cp.array(255, dtype=original.dtype), dtype=original.dtype)
    black_image = cp.zeros_like(original)  # Asegúrate de que black_image tenga 3 canales

    # Ajustar la máscara para que coincida con la imagen original
    if mask.ndim == 2:
        mask = cp.stack([mask] * 3, axis=-1)  # Convertir a 3 canales si es 2D

    # Convertir `mask` a booleano
    mask = mask.astype(cp.bool_)
    mask_invert = cp.invert(mask)

    # Asegurarse de que `original`, `white_image` y `black_image` tengan la misma forma
    if original.ndim == 2:
        original = cp.stack([original] * 3, axis=-1)  # Convertir a 3 canales si es 2D
    if white_image.ndim == 2:
        white_image = cp.stack([white_image] * 3, axis=-1)  # Asegurarse de que white_image tenga 3 canales
    if black_image.ndim == 2:
        black_image = cp.stack([black_image] * 3, axis=-1)  # Asegurarse de que black_image tenga 3 canales

    # Aplicar la máscara
    result = cp.where(mask, white_image, original)
    result = cp.where(mask_invert, black_image, result)

    # Crear una imagen verde donde `mask2` es blanco
    green_color = cp.array([0, 255, 0], dtype=original.dtype)  # Verde en formato RGB
    mask2_colored = cp.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=original.dtype)  # Crear imagen vacía con 3 canales

    # Asignar verde a donde mask2 es blanco
    mask2_colored[mask2 == 255] = green_color

    # Obtener dimensiones y verificar
    if original.ndim == 3:
        height, width, channels = original.shape
    else:
        height, width = original.shape
        channels = 1

    height1, width1 = mask2.shape

    # Verificar dimensiones
    if (height, width) != (height1, width1):
        #print("Redimensionando la máscara para que coincida con las dimensiones de la imagen original.")
        mask2 = cp.asarray(resize_image_bgr(mask2, (width, height)))
        if mask2.ndim == 2:
            mask2_colored = cp.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=original.dtype)
            mask2_colored[mask2 == 255] = green_color
    else:
        #print("Las dimensiones de la máscara ya coinciden con las dimensiones de la imagen original.")
        if mask2.ndim == 2:
            mask2_colored = cp.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=original.dtype)
            mask2_colored[mask2 == 255] = green_color
    """
    # Mostrar las imágenes
    cv2.imshow("original", result.get())
    cv2.imshow("contorno (verde)", mask2_colored.get())
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return result, mask2_colored  # Devuelve la imagen resultante y la máscara ajustada


def adaptive_threshold(image, block_size, C):
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)

    if block_size % 2 == 0:
        raise ValueError("block_size debe ser un número impar.")

    mean_filter = cp.ones((block_size, block_size), dtype=cp.float32) / (block_size * block_size)
    mean_image = cp.signal.convolve(image, mean_filter, mode='same')

    thresholded_image = image - mean_image - C
    thresholded_image = cp.where(thresholded_image > 0, 255, 0).astype(cp.uint8)

    return thresholded_image

def cupy_threshold(image, threshold_value=1, max_value=255):
    # Crea una nueva imagen binaria con el mismo shape que `image`
    binary_mask = cp.zeros_like(image, dtype=cp.uint8)

    # Aplica el umbral: establece en `max_value` los valores mayores que `threshold_value`
    binary_mask[image > threshold_value] = max_value

    return binary_mask
