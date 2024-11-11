import cv2
import numpy as np
import time
import cupy as cp

# Define los colores de segmentación específicos
SEGMENTATION_COLORS = np.array([
    [35, 195, 249],   # Color para agua
    [164, 76, 90],
    [224, 167, 41]     # Color para obstáculo
], np.uint8)

def fuzzify_intensity(value):
    """Fuzzifica la intensidad (canal V en HSV) en categorías bajo, medio y alto."""
    if value < 50:
        return 'low'
    elif 50 <= value < 160:
        return 'medium'
    else:
        return 'high'

def fuzzify_hue_saturation(hue, saturation):
    """Clasifica tonos y saturaciones para identificar agua, cielo u obstáculos."""
    # Clasificación de agua: tonos bajos en saturación media-baja y valor medio
    if 15 <= hue <= 50 and saturation < 30:
        return 'water'
    # Clasificación de cielo: tonos entre 90 y 120 (cian) y alta luminosidad
    elif 90 <= hue <= 120 and saturation < 50:
        return 'sky'
    # Clasificación de obstáculos: tonos cálidos (0-20 o 150-180) con saturación alta
    elif (0 <= hue <= 20 or 90 <= hue <= 120) and saturation > 20:
        return 'obstacle'
    else:
        return 'unknown'

def fuzzify_position(pixel_position, image_height):
    """Fuzzifica la posición del píxel en categorías de parte superior, media e inferior."""
    relative_position = pixel_position / image_height
    if relative_position < 0.33:
        return 'top'
    elif 0.33 <= relative_position < 0.66:
        return 'middle'
    else:
        return 'bottom'

def classify_pixel(intensity_category, color_category, position_category):
    """Clasifica un píxel basado en reglas difusas con valores HSV ajustados."""
    if color_category == 'water' and position_category == 'bottom' and intensity_category == 'high':
        return 'water'
    elif color_category == 'sky' and position_category == 'top' and intensity_category == 'high':
        return 'sky'
    elif color_category == 'obstacle' and (intensity_category == 'medium') and (position_category == 'middle'):
        return 'obstacle'
    elif color_category == 'water' and position_category == 'middle' and intensity_category == "high":
        return 'water'
    elif color_category == 'sky' and position_category == 'middle' and intensity_category == "high":
        return 'sky'
    elif position_category == "bottom" and intensity_category == "high":
        return 'water'
    elif position_category == "top" and intensity_category == "high":
        return 'sky'
    else:
        return 'unknown'

def process_image(image):
    """Procesa una imagen para clasificar cada píxel como agua, cielo u obstáculo."""
    height, width, _ = image.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Definir colores para cada categoría basados en SEGMENTATION_COLORS
    colors = {
        'water': SEGMENTATION_COLORS[2],
        'sky': SEGMENTATION_COLORS[1],
        'obstacle': SEGMENTATION_COLORS[0],
        'unknown': [0, 0, 0]  # Negro para desconocido
    }
    # Procesar cada píxel
    for y in range(height):
        for x in range(width):
            # Convertir de BGR a HSV y obtener los valores HSV
            b, g, r = image[y, x]
            hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            hue, saturation, value = hsv_pixel
            f = (192 - y)/117
            saturation = saturation if f >= 1 else saturation * f

            # Aplicar fuzzificación y clasificación
            intensity_category = fuzzify_intensity(value)
            color_category = fuzzify_hue_saturation(hue, saturation)
            position_category = fuzzify_position(y, height)
            classification = classify_pixel(intensity_category, color_category, position_category)
            output_image[y, x] = colors[classification]  # Asignar color basado en la clasificación

    return output_image

def process_image_cuda(image):
    """Procesa una imagen para clasificar cada píxel como agua, cielo u obstáculo usando CUDA."""
    height, width, _ = image.shape
    output_image = cp.zeros((height, width, 3), dtype=cp.uint8)

    # Definir colores para cada categoría basados en SEGMENTATION_COLORS
    colors = cp.array([
        [35, 195, 249],   # Color para agua
        [164, 76, 90],
        [224, 167, 41],   # Color para obstáculo
        [0, 0, 0]         # Negro para desconocido
    ], dtype=cp.uint8)

    # Convertir la imagen de BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = cp.asarray(hsv_image)

    # Obtener los canales HSV
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]

    # Calcular la posición relativa
    y_coords = cp.arange(height).reshape(-1, 1)
    relative_position = y_coords / height

    # Fuzzificación de intensidad
    intensity_category = cp.where(value < 50, 'low', cp.where(value < 160, 'medium', 'high'))

    # Fuzzificación de tono y saturación
    color_category = cp.where((15 <= hue) & (hue <= 50) & (saturation < 30), 'water',
                    cp.where((90 <= hue) & (hue <= 120) & (saturation < 50), 'sky',
                    cp.where(((0 <= hue) & (hue <= 20) | (90 <= hue) & (hue <= 120)) & (saturation > 20), 'obstacle', 'unknown')))

    # Fuzzificación de posición
    position_category = cp.where(relative_position < 0.33, 'top', cp.where(relative_position < 0.66, 'middle', 'bottom'))

    # Clasificación de píxeles
    classification = cp.where((color_category == 'water') & (position_category == 'bottom') & (intensity_category == 'high'), 'water',
                    cp.where((color_category == 'sky') & (position_category == 'top') & (intensity_category == 'high'), 'sky',
                    cp.where((color_category == 'obstacle') & (intensity_category == 'medium') & (position_category == 'middle'), 'obstacle',
                    cp.where((color_category == 'water') & (position_category == 'middle') & (intensity_category == 'high'), 'water',
                    cp.where((color_category == 'sky') & (position_category == 'middle') & (intensity_category == 'high'), 'sky',
                    cp.where((position_category == 'bottom') & (intensity_category == 'high'), 'water',
                    cp.where((position_category == 'top') & (intensity_category == 'high'), 'sky', 'unknown')))))))

    # Asignar colores basados en la clasificación
    output_image = colors[classification]

    return cp.asnumpy(output_image)

if __name__ == "__main__":
    # Leer y redimensionar la imagen
    tic = time.time()
    tics = {}
    for i in range(3):
        tics[f"image_{i+1}"] = []
        # for j in range(100):
        tic1 = time.time()
        image_path = f'/home/tille/Desktop/Tesis/WaSR-T/images/akaso{i+1}.jpeg'  # Ruta de la imagen de ejemplo
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (256, 192))

        # Procesar la imagen redimensionada
        classified_image = process_image(resized_image)

        # Guardar y mostrar el resultado
        output_path = f'/home/tille/Desktop/Tesis/WaSR-T/images/akaso{i+1}_fuzzy.png'
        cv2.imwrite(output_path, classified_image)
        tics[f"image_{i+1}"].append(time.time()-tic1)
        # print(f"Imagen {i+1} procesada {j+1} veces.")

    # with open('times_fuzzy.pickle', 'wb') as f:
    #     pickle.dump(tics, f)