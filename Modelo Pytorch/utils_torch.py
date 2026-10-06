import torch

def crop_horizontal(imagen, indice_vertical):
    """
    Recorta una imagen a color horizontalmente en un índice dado usando PyTorch.
    
    Args:
        imagen: Una imagen a color en formato PyTorch (tensor).
        indice_vertical: El índice vertical donde se realizará el recorte.
    
    Returns:
        Una tupla que contiene dos imágenes: la parte superior y la parte inferior.
    """

    # Verifica si indice_vertical es un tensor de más de un elemento
    if isinstance(indice_vertical, torch.Tensor):
        if indice_vertical.numel() > 1:
            raise ValueError("indice_vertical debe ser un valor único, pero se recibió un tensor de más de un elemento.")
        indice_vertical = indice_vertical.item()

    if indice_vertical < 0 or indice_vertical >= imagen.shape[0]:
        raise ValueError("El índice vertical está fuera de los límites de la imagen.")
    
    if imagen.ndim == 3:
        parte_superior = imagen[:indice_vertical, :, :]
        parte_inferior = imagen[indice_vertical:, :, :]
    else:
        parte_superior = imagen[:indice_vertical, :]
        parte_inferior = imagen[indice_vertical:, :]
    
    return parte_superior, parte_inferior


def segment_and_identify_objects(image_gray, mask_binary, original, block_size=15, threshold_area=60):
    # Paso 2: Analizar bloques de 15x15 píxeles
    height, width = image_gray.shape[:2]
    # Generar índices de los bloques
    y_indices = torch.arange(block_size, height - block_size, block_size)
    x_indices = torch.arange(block_size, width - block_size, block_size)
    
    # Crear grids de coordenadas
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Extraer los bloques de la imagen y la máscara utilizando list comprehension
    block_images = torch.stack([image_gray[y:y + block_size, x:x + block_size] for y, x in zip(y_grid.flatten(), x_grid.flatten())])
    block_masks = torch.stack([mask_binary[y:y + block_size, x:x + block_size] for y, x in zip(y_grid.flatten(), x_grid.flatten())])
    
    # Contar los píxeles negros en los bloques
    black_pixel_counts = torch.sum(torch.all(block_images == 0, dim=-1), dim=(1, 2))
    
    # Identificar píxeles verdes en los bloques
    green_pixels = (block_masks[:, :, :, 1] > 100) & (block_masks[:, :, :, 0] < 50) & (block_masks[:, :, :, 2] < 50)
    green_pixel_counts = torch.sum(green_pixels, dim=(1, 2))
    
    # Crear un vector booleano que indique qué bloques cumplen la condición inicial
    valid_blocks = (black_pixel_counts > threshold_area) & (green_pixel_counts <= 30)
    
    # Verificar bloques adyacentes
    rects = []
    for idx, (y, x) in enumerate(zip(y_grid.flatten(), x_grid.flatten())):
        if not valid_blocks[idx]:
            continue

        # Verificar bloques adyacentes
        adyacente_verificado = False

        # Verificar los 3 bloques adyacentes a la derecha (i, j+1), (i+1, j+1), (i-1, j+1)
        if x + block_size < width and y + block_size < height and y - block_size >= 0:
            block_mask_right1 = mask_binary[y:y + block_size, x + block_size:x + 2 * block_size]
            block_mask_right2 = mask_binary[y + block_size:y + 2 * block_size, x + block_size:x + 2 * block_size]
            block_mask_right3 = mask_binary[y - block_size:y, x + block_size:x + 2 * block_size]
            green_pixels_right1 = (block_mask_right1[:, :, 1] > 100) & (block_mask_right1[:, :, 0] < 50) & (block_mask_right1[:, :, 2] < 50)
            green_pixels_right2 = (block_mask_right2[:, :, 1] > 100) & (block_mask_right2[:, :, 0] < 50) & (block_mask_right2[:, :, 2] < 50)
            green_pixels_right3 = (block_mask_right3[:, :, 1] > 100) & (block_mask_right3[:, :, 0] < 50) & (block_mask_right3[:, :, 2] < 50)
            if torch.sum(green_pixels_right1) > 20 and torch.sum(green_pixels_right2) > 20 and torch.sum(green_pixels_right3) > 20:
                adyacente_verificado = True
        
        # Verificar los 3 bloques adyacentes a la izquierda
        if x - block_size >= 0 and y + block_size < height and y - block_size >= 0:
            block_mask_left1 = mask_binary[y:y + block_size, x - block_size:x]
            block_mask_left2 = mask_binary[y + block_size:y + 2 * block_size, x - block_size:x]
            block_mask_left3 = mask_binary[y - block_size:y, x - block_size:x]
            green_pixels_left1 = (block_mask_left1[:, :, 1] > 100) & (block_mask_left1[:, :, 0] < 50) & (block_mask_left1[:, :, 2] < 50)
            green_pixels_left2 = (block_mask_left2[:, :, 1] > 100) & (block_mask_left2[:, :, 0] < 50) & (block_mask_left2[:, :, 2] < 50)
            green_pixels_left3 = (block_mask_left3[:, :, 1] > 100) & (block_mask_left3[:, :, 0] < 50) & (block_mask_left3[:, :, 2] < 50)
            if torch.sum(green_pixels_left1) > 20 and torch.sum(green_pixels_left2) > 20 and torch.sum(green_pixels_left3) > 20:
                adyacente_verificado = True
        
        # Verificar los 3 bloques adyacentes hacia abajo
        if y + block_size < height and x + block_size < width and x - block_size >= 0:
            block_mask_down1 = mask_binary[y + block_size:y + 2 * block_size, x:x + block_size]
            block_mask_down2 = mask_binary[y + block_size:y + 2 * block_size, x + block_size:x + 2 * block_size]
            block_mask_down3 = mask_binary[y + block_size:y + 2 * block_size, x - block_size:x]
            green_pixels_down1 = (block_mask_down1[:, :, 1] > 100) & (block_mask_down1[:, :, 0] < 50) & (block_mask_down1[:, :, 2] < 50)
            green_pixels_down2 = (block_mask_down2[:, :, 1] > 100) & (block_mask_down2[:, :, 0] < 50) & (block_mask_down2[:, :, 2] < 50)
            green_pixels_down3 = (block_mask_down3[:, :, 1] > 100) & (block_mask_down3[:, :, 0] < 50) & (block_mask_down3[:, :, 2] < 50)
            if torch.sum(green_pixels_down1) > 20 and torch.sum(green_pixels_down2) > 20 and torch.sum(green_pixels_down3) > 20:
                adyacente_verificado = True
        
        # Verificar los 3 bloques adyacentes hacia arriba
        if y - block_size >= 0 and x + block_size < width and x - block_size >= 0:
            block_mask_up1 = mask_binary[y - block_size:y, x:x + block_size]
            block_mask_up2 = mask_binary[y - block_size:y, x + block_size:x + 2 * block_size]
            block_mask_up3 = mask_binary[y - block_size:y, x - block_size:x]
            green_pixels_up1 = (block_mask_up1[:, :, 1] > 100) & (block_mask_up1[:, :, 0] < 50) & (block_mask_up1[:, :, 2] < 50)
            green_pixels_up2 = (block_mask_up2[:, :, 1] > 100) & (block_mask_up2[:, :, 0] < 50) & (block_mask_up2[:, :, 2] < 50)
            green_pixels_up3 = (block_mask_up3[:, :, 1] > 100) & (block_mask_up3[:, :, 0] < 50) & (block_mask_up3[:, :, 2] < 50)
            if torch.sum(green_pixels_up1) > 20 and torch.sum(green_pixels_up2) > 20 and torch.sum(green_pixels_up3) > 20:
                adyacente_verificado = True
        
        # Si el bloque es válido y tiene bloques adyacentes verdes, agregar a rects
        if valid_blocks[idx] and adyacente_verificado:
            rects.append((x.item(), y.item(), block_size, block_size))  # Guardar coordenadas del bloque

    return rects
            
import torch

def resize_image_bgr(image, new_shape):
    """
    Redimensiona una imagen en formato BGR a una nueva forma especificada.

    Parameters
    ----------
    image : torch.Tensor
        Imagen a redimensionar, con forma (altura, anchura, canales).
    new_shape : tuple
        Nueva forma de la imagen en formato (nueva_altura, nueva_anchura).

    Returns
    -------
    torch.Tensor
        Imagen redimensionada en formato BGR.
    """
    new_height, new_width = new_shape
    # image: (H, W, C) -> (1, C, H, W)
    img_perm = image.permute(2, 0, 1).unsqueeze(0).float()
    resized = torch.nn.functional.interpolate(
        img_perm, size=(new_height, new_width), mode='bilinear', align_corners=False
    )
    # (1, C, H, W) -> (H, W, C)
    return resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(image.dtype)


def hacer_mascara2(image3, mask2):
    """
    Aplica una máscara a la imagen y crea una versión coloreada de la máscara.

    Parameters
    ----------
    image3 : torch.Tensor
        Imagen en formato BGR (o escala de grises) con forma (altura, anchura, canales).
    mask2 : torch.Tensor
        Máscara binaria que indica áreas a colorear, con forma (altura, anchura).

    Returns
    -------
    torch.Tensor
        Imagen resultante con la máscara aplicada.
    """
    
    # Asegúrate de que la imagen está en formato de tensor de PyTorch
    image3_tensor = image3.clone()

    # Verifica si la imagen tiene 3 canales
    if image3_tensor.ndim == 3:
        original = image3_tensor[:, :, ::-1]  # Convertir de BGR a RGB
    else:
        original = image3_tensor  # No es necesario cambiar si es escala de grises

    # Crear la máscara basada en el rango de valores
    if original.ndim == 3:
        mask = (torch.all(original >= 0, dim=-1) & torch.all(original <= 15, dim=-1)).to(torch.uint8) * 255
    else:
        mask = ((original >= 0) & (original <= 15)).to(torch.uint8) * 255

    # Crear imágenes blancas y negras
    white_image = torch.full(original.shape, torch.tensor(255, dtype=original.dtype), dtype=original.dtype)
    black_image = torch.zeros_like(original)  # Asegúrate de que black_image tenga 3 canales

    # Ajustar la máscara para que coincida con la imagen original
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1).expand(-1, -1, 3)  # Convertir a 3 canales si es 2D

    # Convertir `mask` a booleano
    mask = mask.bool()
    mask_invert = ~mask

    # Asegurarse de que `original`, `white_image` y `black_image` tengan la misma forma
    if original.ndim == 2:
        original = original.unsqueeze(-1).expand(-1, -1, 3)  # Convertir a 3 canales si es 2D
    if white_image.ndim == 2:
        white_image = white_image.unsqueeze(-1).expand(-1, -1, 3)  # Asegurarse de que white_image tenga 3 canales
    if black_image.ndim == 2:
        black_image = black_image.unsqueeze(-1).expand(-1, -1, 3)  # Asegurarse de que black_image tenga 3 canales

    # Aplicar la máscara
    result = torch.where(mask, white_image, original)
    result = torch.where(mask_invert, black_image, result)

    # Crear una imagen verde donde `mask2` es blanco
    green_color = torch.tensor([0, 255, 0], dtype=original.dtype)  # Verde en formato RGB
    mask2_colored = torch.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=original.dtype)  # Crear imagen vacía con 3 canales

    # Asignar verde a donde mask2 es blanco
    mask2_colored[mask2 != 0] = green_color

    # Obtener dimensiones y verificar
    if original.ndim == 3:
        height, width, channels = original.shape
    else:
        height, width = original.shape
        channels = 1

    height1, width1 = mask2.shape

    # Verificar dimensiones (opcional)
    if (height != height1) or (width != width1):
        raise ValueError("Las dimensiones de la imagen y la máscara no coinciden.")
    
    return result, mask2_colored

import torch

def pytorch_threshold(image, threshold_value=1, max_value=255):
    """
    Aplica un umbral a la imagen y devuelve una máscara binaria.

    Parameters
    ----------
    image : torch.Tensor
        Imagen de entrada, se espera que sea un tensor 2D o 3D.
    threshold_value : float, optional
        Valor umbral para la binarización. Por defecto es 1.
    max_value : int, optional
        Valor asignado a los píxeles por encima del umbral. Por defecto es 255.

    Returns
    -------
    torch.Tensor
        Máscara binaria de la misma forma que la imagen original.
    """
    # Crea una nueva imagen binaria con el mismo shape que `image`
    binary_mask = torch.zeros_like(image, dtype=torch.uint8)

    # Aplica el umbral: establece en `max_value` los valores mayores que `threshold_value`
    binary_mask[image > threshold_value] = max_value

    return binary_mask

