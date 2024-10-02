import cupy as cp
import cv2
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
    black_pixel_counts = cp.sum(cp.all(block_images > 200, axis=-1), axis=(1, 2))
    
    # Identificar píxeles verdes en los bloques
    green_pixels = (block_masks[:, :, :, 1] > 100) & (block_masks[:, :, :, 0] < 50) & (block_masks[:, :, :, 2] < 50)
    green_pixel_counts = cp.sum(green_pixels, axis=(1, 2))
    print(green_pixel_counts)
    # Crear un vector booleano que indique qué bloques cumplen la condición inicial
    valid_blocks = (black_pixel_counts > threshold_area) & (green_pixel_counts <= 20)
    print(black_pixel_counts)
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
        
        # Si cualquiera de las direcciones tiene 3 bloques adyacentes con suficientes píxeles verdes
        if adyacente_verificado:
            rects.append((x, y, block_size, block_size))
    
    n = len(rects)
    print(n)
    
    # Función para marcar rectángulos en rojo
    def mark_rect_red(rect):
        x, y, w, h = rect
        original[y:y+h, x:x+w, :] = cp.array([0, 0, 255])  # Rojo

    # Marcar el resto de los rectángulos en rojo
    for rect in rects:
        x, y, w, h = rect
        color = original[y, x]
        if cp.all(color == cp.array([0, 0, 255])) or cp.all(color == cp.array([0, 255, 255])) or cp.all(color == cp.array([255, 0, 0])):
            continue
        else:
            mark_rect_red(rect)
    
    # Convertir a un formato que pueda mostrar la imagen
    image_to_show = cp.asnumpy(original)

    # Mostrar la imagen con los bloques marcados
    cv2.imshow('Segmented Image with Detected Objects', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

def hacer_mascara(image3, fila_interes):
    # Leer la máscara (suponiendo que ya está en la GPU como array de CuPy)
    mask = cp.array(cv2.imread("imagen_umbral.png", cv2.IMREAD_GRAYSCALE))

    # Obtener las dimensiones de la máscara
    x, y = mask.shape

    # Asumiendo que `crop_horizontal` también está en CuPy y devuelve un array de CuPy
    _, mask2 = crop_horizontal(mask, fila_interes)
    print(image3.shape)

    # Leer la imagen original (suponiendo que ya está en la GPU como array de CuPy)
    image3_numpy = cp.asnumpy(image3)

# Aplicar cv2.cvtColor en NumPy
    image3_rgb_numpy = cv2.cvtColor(image3_numpy, cv2.COLOR_BGR2RGB)

# Convertir de nuevo a CuPy si es necesario
    original = cp.array(image3_rgb_numpy)

    # Crear una máscara para los píxeles negros
    mask = cp.logical_and(
        cp.all(original >= cp.array([0, 0, 0]), axis=-1),
        cp.all(original <= cp.array([15, 15, 15]), axis=-1)
    ).astype(cp.uint8) * 255

    # Crear imágenes en negro y blanco
    white_image = cp.full_like(original, cp.array(255, dtype=original.dtype))

    black_image = cp.zeros_like(original)

    if mask.ndim == 2:
        mask = cp.stack([mask] * 3, axis=-1)  # Convertir a 3 canales si es necesario
    # Aplicar la máscara para obtener la imagen con negros convertidos a blancos y el resto a negro
        # Convertir `mask` a un array booleano para la operación bitwise
    mask = mask.astype(cp.bool_)
    mask_invert = cp.invert(mask)
     # Aplicar la máscara usando operaciones condicionales
    result = cp.where(mask, white_image, original)
    result = cp.where(mask_invert, black_image, result)
    # Obtener las dimensiones de la imagen original y la máscara recortada
    height, width, channels = original.shape
    height1, width1 = mask2.shape

    # Verificar las dimensiones de ambas imágenes
    if (height, width) != (height1, width1):
        print("Redimensionando la máscara para que coincida con las dimensiones de la imagen original.")
        mask2 = cp.array(cv2.resize(cp.asnumpy(mask2), (width, height)))
    else:
        print("Las dimensiones de la máscara ya coinciden con las dimensiones de la imagen original.")

    # Asegurarse de que la máscara sea binaria
    _, mask_binary = cv2.threshold(cp.asnumpy(mask2), 1, 255, cv2.THRESH_BINARY)
    mask_binary = cp.array(mask_binary)

    # Crear una imagen de contornos verdes (tamaño de la imagen original)
    contour_image = cp.zeros((original.shape[0], original.shape[1], 3), dtype=cp.uint8)

    # Establecer los contornos en verde (BGR) usando la máscara binaria
    contour_image[mask_binary > 0] = [0, 255, 0]  # Verde en BGR

    # Crear una imagen RGBA con el fondo transparente
    rgba_image = cp.zeros((original.shape[0], original.shape[1], 4), dtype=cp.uint8)

    # Copiar la imagen original al canal RGB de la imagen RGBA
    rgba_image[:, :, :3] = original

    # Crear una máscara para el canal alfa (transparencia)
    transparency_mask = cp.zeros((original.shape[0], original.shape[1]), dtype=cp.uint8)
    transparency_mask[mask_binary > 0] = 255  # Píxeles de contorno serán completamente opacos

    # Copiar la máscara de transparencia al canal alfa de la imagen RGBA
    rgba_image[:, :, 3] = transparency_mask

    # Aplicar la imagen de contornos al canal alfa de la imagen RGBA
    # Para hacer que los contornos sean visibles, combinamos la imagen original con la imagen de contornos
    combined_image = original + contour_image
    # Iterar sobre cada píxel y mostrar sus valores en los tres canales
    """for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            r, g, b = original[i, j]
            print(f"Píxel ({i}, {j}) - R: {r}, G: {g}, B: {b}")"""


    #cv2.imshow("original", rgba_image.get())
    #cv2.imshow("contorno", contour_image.get())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(original.shape)
    print(contour_image.shape)
    return original, contour_image

