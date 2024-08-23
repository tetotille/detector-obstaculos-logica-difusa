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

    if indice_vertical < 0 or indice_vertical >= imagen.shape[0]:
        raise ValueError("El índice vertical está fuera de los límites de la imagen.")
    
    parte_superior = imagen[:indice_vertical, :]
    parte_inferior = imagen[indice_vertical:, :]
    
    return parte_superior, parte_inferior

def segment_and_identify_objects(image_gray, mask_binary, original, block_size=15, threshold_area=155):
    # Paso 2: Analizar bloques de 15x15 píxeles
    height, width, channels = image_gray.shape
    rects = []  # Lista para almacenar los rectángulos detectados
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extraer el bloque de la imagen y de la máscara
            block_image = image_gray[y:y+block_size, x:x+block_size]
            block_mask = mask_binary[y:y+block_size, x:x+block_size]

            # Contar los píxeles negros en el bloque de la imagen en todos los canales
            black_pixel_count = cp.sum(cp.all(block_image == cp.array([0, 0, 0]), axis=-1))

            # Verde: canal verde alto y canales rojo y azul bajos
            green_pixels = (block_mask[:, :, 1] > 100) & (block_mask[:, :, 0] < 50) & (block_mask[:, :, 2] < 50)
            green_pixel_count = cp.sum(green_pixels)

            # Si el bloque contiene suficientes píxeles negros y tiene contorno en la máscara, marcar el bloque
            if black_pixel_count > threshold_area and green_pixel_count > 70:
                rects.append((x, y, block_size, block_size))  # Almacena el rectángulo

    # Paso 3: Detectar y marcar rectángulos alineados horizontalmente que cubren toda la fila
    blocks_per_row = int(width // block_size)  # Número de bloques que caben en una fila
    blocks_per_column = int(height // block_size)
    rects_by_column = {}
    rects_by_row = {}
    
    for rect in rects:
        x, y, w, h = rect
        if x not in rects_by_column:
            rects_by_column[x] = []
        rects_by_column[x].append(rect)
        
    for rect1 in rects:
        x, y, w, h = rect1
        if y not in rects_by_row:
            rects_by_row[y] = []
        rects_by_row[y].append(rect1)

    # Función para marcar filas en azul
    def mark_row_blue(row_rects):
        for rect in row_rects:
            x, y, w, h = rect
            original[y:y+h, x:x+w, :] = cp.array([255, 0, 0])  # Azul

    # Función para marcar columnas en amarillo
    def mark_column_yellow(column_rects):
        for rect in column_rects:
            x, y, w, h = rect
            original[y:y+h, x:x+w, :] = cp.array([0, 255, 255])  # Amarillo

    # Función para marcar rectángulos en rojo
    def mark_rect_red(rect):
        x, y, w, h = rect
        original[y:y+h, x:x+w, :] = cp.array([0, 0, 255])  # Rojo

    # Verificar y marcar filas completas
    for y, row_rects in rects_by_row.items():
        row_rects.sort()
        if len(row_rects) == blocks_per_row:
            first_rect_x = row_rects[0][0]
            last_rect_x = row_rects[-1][0] + row_rects[-1][2]
            mark_row_blue(row_rects)

            # Verificar y marcar la fila superior si está alineada
            top_y = y - row_rects[0][3]
            if top_y in rects_by_row:
                top_row_rects = rects_by_row[top_y]
                top_first_x = top_row_rects[0][0]
                top_last_x = top_row_rects[-1][0] + top_row_rects[-1][2]
                if (top_first_x <= first_rect_x <= top_last_x) or (top_first_x <= last_rect_x <= top_last_x):
                    mark_row_blue(top_row_rects)

            # Verificar y marcar la fila inferior si está alineada
            bottom_y = y + row_rects[0][3]
            if bottom_y in rects_by_row:
                bottom_row_rects = rects_by_row[bottom_y]
                bottom_first_x = bottom_row_rects[0][0]
                bottom_last_x = bottom_row_rects[-1][0] + bottom_row_rects[-1][2]
                if (bottom_first_x <= first_rect_x <= bottom_last_x) or (bottom_first_x <= last_rect_x <= bottom_last_x):
                    mark_row_blue(bottom_row_rects)

    # Verificar y marcar columnas completas
    for column_x, column_rects in rects_by_column.items():
        if len(column_rects) == blocks_per_column:
            mark_column_yellow(column_rects)

    # Marcar el resto de los rectángulos en rojo
    for rect in rects:
        x, y, w, h = rect
        color = original[y, x]
        if cp.all(color == cp.array([0, 0, 255])) or cp.all(color == cp.array([0, 255, 255])) or cp.all(color == cp.array([255, 0, 0])):
            continue
        else:
            mark_rect_red(rect)

    # Verificar y marcar columnas completas (incluso si ya están marcadas en azul)
    for column_x, column_rects in rects_by_column.items():
        if len(column_rects) == blocks_per_column:
            for rect in column_rects:
                x, y, w, h = rect
                color = original[y, x]
                if cp.all(color == cp.array([255, 0, 0])):  # Si está marcado en azul
                    original[y:y+h, x:x+w, :] = cp.array([0, 255, 255])  # Amarillo

    # Convertir a un formato que pueda mostrar la imagen
    image_to_show = cp.asnumpy(original)

    # Mostrar la imagen con los bloques marcados
    cv2.imshow('Segmented Image with Detected Objects', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image, new_shape):
    # Obtén las dimensiones originales y las nuevas dimensiones
    orig_shape = image.shape
    new_height, new_width = new_shape

    # Crea matrices para las nuevas coordenadas
    y = cp.linspace(0, orig_shape[0] - 1, new_height)
    x = cp.linspace(0, orig_shape[1] - 1, new_width)
    x_grid, y_grid = cp.meshgrid(x, y)

    # Interpolación bilineal
    x0 = cp.floor(x_grid).astype(cp.int32)
    x1 = x0 + 1
    y0 = cp.floor(y_grid).astype(cp.int32)
    y1 = y0 + 1

    x0 = cp.clip(x0, 0, orig_shape[1] - 1)
    x1 = cp.clip(x1, 0, orig_shape[1] - 1)
    y0 = cp.clip(y0, 0, orig_shape[0] - 1)
    y1 = cp.clip(y1, 0, orig_shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x_grid) * (y1 - y_grid)
    wb = (x1 - x_grid) * (y_grid - y0)
    wc = (x_grid - x0) * (y1 - y_grid)
    wd = (x_grid - x0) * (y_grid - y0)

    resized_image = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # Asegurarse de que los valores estén dentro del rango [0, 255]
    resized_image = cp.clip(resized_image, 0, 255)

    return resized_image
# Usar la función con la ruta de la imagen, la máscara y la imagen original en forma de array de CuPy
