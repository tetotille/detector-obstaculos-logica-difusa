import cv2
import time
from utils import neighbor_framed_np
import numpy as cp

# Define los colores de segmentación específicos
SEGMENTATION_COLORS = cp.array([
    (0,0,0), # desconocido
    (35, 195, 249),   # Color para agua
    (224, 167, 41)     # Color para obstáculo
], cp.uint8)


def view_images(image):
    import matplotlib.pyplot as plt

    # Convertir la imagen de BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Separar los canales HSV
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    # Separar los canales RGB
    b_channel, g_channel, r_channel = cv2.split(image)

    # Mostrar los canales HSV y RGB por separado
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(h_channel, cmap='gray')
    axs[0, 0].set_title('Hue Channel')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(s_channel, cmap='gray')
    axs[0, 1].set_title('Saturation Channel')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(v_channel, cmap='gray')
    axs[0, 2].set_title('Value Channel')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(r_channel, cmap='gray')
    axs[1, 0].set_title('Red Channel')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(g_channel, cmap='gray')
    axs[1, 1].set_title('Green Channel')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(b_channel, cmap='gray')
    axs[1, 2].set_title('Blue Channel')
    axs[1, 2].axis('off')

    plt.show()

def blue_intensity(b,mode):
    """Fuzzifica la intensidad del canal azul."""
    if b < mode-70:
        return 'low'
    elif b < mode:
        return 'medium'
    else:
        return 'high'
    
def red_intensity(r,mode):
    """Fuzzifica la intensidad del canal rojo."""
    if r < mode-20:
        return 'low'
    elif r < mode:
        return 'medium'
    else:
        return 'high'
    
def green_intensity(g,mode):
    """Fuzzifica la intensidad del canal verde."""
    if g < mode-50:
        return 'low'
    elif g < mode:
        return 'medium'
    else:
        return 'high'
    
def blue_intensity_vectorized(b, mode):
    """Fuzzifica la intensidad del canal azul en bloques."""
    categories = cp.zeros(b.shape, dtype=cp.int8)  # Inicializar como 'low' (0)
    categories[b >= max(mode-130,60)] = 1  # Asignar 'medium' (1)
    categories[b >= min(mode - 30,193)] = 2  # Asignar 'high' (2)
    return categories

def red_intensity_vectorized(r, mode):
    """Fuzzifica la intensidad del canal rojo en bloques."""
    categories = cp.zeros(r.shape, dtype=cp.int8)  # Inicializar como 'low' (0)
    categories[r >= max(mode-130,60)] = 1  # Asignar 'medium' (1)
    categories[r >= min(mode - 30,193)] = 2  # Asignar 'high' (2)
    return categories

def green_intensity_vectorized(g, mode):
    """Fuzzifica la intensidad del canal verde en bloques."""
    categories = cp.zeros(g.shape, dtype=cp.int8)  # Inicializar como 'low' (0)
    categories[g >= max(mode-130,60)] = 1  # Asignar 'medium' (1)
    categories[g >= min(mode - 30,193)] = 2  # Asignar 'high' (2)
    return categories


def classify_pixel(red_category, blue_category, green_category):
    """Clasifica un píxel basado en reglas difusas con valores HSV ajustados."""
    if red_category == "low" and blue_category == "low" and green_category == "low":
        return 'water'
    elif red_category == "medium" and blue_category == "medium" and green_category == "medium":
        return 'water'
    elif red_category == "high" and blue_category == "high" and green_category == "high":
        return 'water'
    elif red_category == "high" and blue_category == "high" and green_category == "medium":
        return 'water'
    elif red_category == "high" and blue_category == "medium" and green_category == "high":
        return 'water'
    elif red_category == "medium" and blue_category == "high" and green_category == "high":
        return 'water'
    elif red_category == "low" and blue_category == "low" and green_category == "high":
        return 'obstacle'
    elif red_category == "low" and blue_category == "high" and green_category == "low":
        return 'obstacle'
    elif red_category == "high" and blue_category == "low" and green_category == "low":
        return 'obstacle'
    elif red_category == "low" and blue_category == "low" and green_category == "medium":
        return 'obstacle'
    elif red_category == "low" and blue_category == "medium" and green_category == "low":
        return 'obstacle'
    elif red_category == "medium" and blue_category == "low" and green_category == "low":
        return 'obstacle'
    elif red_category == "low" and blue_category == "high" and green_category == "high":
        return 'obstacle'
    elif red_category == "high" and blue_category == "high" and green_category == "low":
        return 'obstacle'
    elif red_category == "high" and blue_category == "low" and green_category == "high":
        return 'obstacle'
    else:
        return 'unknown'
    
def classify_pixel_vectorized(red_category, blue_category, green_category):
    """Clasifica píxeles en bloques vectorizados basado en reglas difusas."""
    # Inicializar clasificación como 'unknown'
    classification = cp.full(red_category.shape, 0)

    # Aplicar reglas difusas en paralelo
    obstacle_mask = (
        (red_category == 0) & (blue_category == 0) & (green_category == 2)
    ) | (
        (red_category == 0) & (blue_category == 2) & (green_category == 0)
    ) | (
        (red_category == 2) & (blue_category == 0) & (green_category == 0)
    ) | (
        (red_category == 0) & (blue_category == 0) & (green_category == 1)
    ) | (
        (red_category == 0) & (blue_category == 1) & (green_category == 0)
    ) | (
        (red_category == 1) & (blue_category == 0) & (green_category == 0)
    ) | (
        (red_category == 0) & (blue_category == 2) & (green_category == 2)
    ) | (
        (red_category == 2) & (blue_category == 2) & (green_category == 0)
    ) | (
        (red_category == 2) & (blue_category == 0) & (green_category == 2)
    )

    water_mask = (
        (red_category == 0) & (blue_category == 0) & (green_category == 0)
    ) | (
        (red_category == 1) & (blue_category == 1) & (green_category == 1)
    ) | (
        (red_category == 2) & (blue_category == 2) & (green_category == 2)
    ) | (
        (red_category == 2) & (blue_category == 1) & (green_category == 2)
    ) | (
        (red_category == 1) & (blue_category == 2) & (green_category == 2)
    )

    # Asignar clasificaciones
    classification[obstacle_mask] = 1
    classification[water_mask] = 2

    return classification

def process_image_cpu(image):
    """Procesa una imagen para clasificar cada píxel como agua, cielo u obstáculo, optimizando con NumPy."""

    height, width, _ = image.shape
    
    # Calcular modos de los canales (sin los valores extremos)
    b_mode = cp.bincount(image[:, :, 0][image[:, :, 0] != 0].flatten()).argmax()
    g_mode = cp.bincount(image[:, :, 1][image[:, :, 1] != 255].flatten()).argmax()
    r_mode = cp.bincount(image[:, :, 2][image[:, :, 2] != 0].flatten()).argmax()

    # Fuzzificación y clasificación vectorizada
    red_categories = red_intensity_vectorized(image[:, :, 2], r_mode)
    green_categories = green_intensity_vectorized(image[:, :, 1], g_mode)
    blue_categories = blue_intensity_vectorized(image[:, :, 0], b_mode)

    # Clasificación por píxel
    classifications = classify_pixel_vectorized(red_categories, blue_categories, green_categories)

    # Mapear resultados a colores y binarios
    output_image = SEGMENTATION_COLORS[classifications]
    binary_image = cp.zeros((height, width), dtype=cp.uint8)
    binary_image[classifications == 1] = 1

    # Obtener cuadros basados en vecinos
    cuadros = neighbor_framed_np(binary_image)

    return output_image, cuadros

def process_image_gpu(image):
    # Fallback to CPU if requested or if GPU fails
    return process_image_cpu(image)

def get_orientation(image:cp.array):
    # Convertir la imagen a un formato binario basado en el color del obstáculo
    obstacle_color = cp.array([224, 167, 41], dtype=cp.uint8)
    mask = cv2.inRange(image, obstacle_color, obstacle_color)
    
    # Encontrar contornos en la máscara binaria
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_coords = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        obstacle_coords.append((x, y, w, h))

    # Determinar la posición de los obstáculos
    positions = []
    for (x, y, w, h) in obstacle_coords:
        center_x = x + w // 2
        if center_x < image.shape[1] // 3:
            positions.append("left")
        elif center_x < 2 * image.shape[1] // 3:
            positions.append("center")
        else:
            positions.append("right")

    return obstacle_coords, positions

if __name__ == "__main__":
    # Leer y redimensionar la imagen
    tic = time.time()
    tics = {}
    for i in range(3):
        tics[f"image_{i+1}"] = []
        tic1 = time.time()
        image_path = f'assets/images/akaso{i+1}.jpeg' 
        image = cv2.imread(image_path)
        if image is None: continue
        resized_image = cv2.resize(image, (256, 192))

        # Procesar la imagen redimensionada
        classified_image, _ = process_image_cpu(resized_image)

        # Guardar y mostrar el resultado
        output_path = f'main_output/output_{i+1}.png'
        os.makedirs("main_output", exist_ok=True)
        cv2.imwrite(output_path, classified_image)
        tics[f"image_{i+1}"].append(time.time()-tic1)
