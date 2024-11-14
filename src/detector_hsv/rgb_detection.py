import cv2
import numpy as np
import time

try:
    import cupy as cp
except:
    print("cuda no está instalado.")
    import numpy as cp

# Define los colores de segmentación específicos
SEGMENTATION_COLORS = np.array([
    [35, 195, 249],   # Color para agua
    # [164, 76, 90],
    [224, 167, 41]     # Color para obstáculo
], np.uint8)


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

def classify_pixel(red_category, blue_category, green_category):
    """Clasifica un píxel basado en reglas difusas con valores HSV ajustados."""
    if red_category == "low" and blue_category == "low" and green_category == "low":
        return 'obstacle'
    elif red_category == "high" and blue_category == "high" and green_category == "high":
        return 'water'
    elif red_category == "high" and blue_category == "high" and green_category == "medium":
        return 'water'
    elif red_category == "high" and blue_category == "medium" and green_category == "high":
        return 'water'
    elif red_category == "medium" and blue_category == "high" and green_category == "high":
        return 'water'
    elif red_category == "low" and blue_category == "low" and green_category == "medium":
        return 'obstacle'
    elif red_category == "low" and blue_category == "medium" and green_category == "low":
        return 'obstacle'
    elif red_category == "medium" and blue_category == "low" and green_category == "low":
        return 'obstacle'
    else:
        return 'unknown'

def process_image(image):
    """Procesa una imagen para clasificar cada píxel como agua, cielo u obstáculo."""
    height, width, _ = image.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    binary_image = np.zeros((height, width), dtype=np.uint8)

    # Definir colores para cada categoría basados en SEGMENTATION_COLORS
    colors = {
        'water': SEGMENTATION_COLORS[1],
        # 'sky': SEGMENTATION_COLORS[1],
        'obstacle': SEGMENTATION_COLORS[0],
        'unknown': [0, 0, 0]  # Negro para desconocido
    }

    # # TEST ONLY # #
    # view_images(image)
    #################
    image_flatten = image.flatten()
    b_mode = np.bincount(image_flatten[0::3][image_flatten[0::3] != 0]).argmax()
    g_mode = np.bincount(image_flatten[1::3][image_flatten[1::3] != 255]).argmax()
    r_mode = np.bincount(image_flatten[2::3][image_flatten[2::3] != 0]).argmax()
    
    print("blue:",b_mode)
    print("green:",g_mode)
    print("red:",r_mode)

    # Procesar cada píxel
    for y in range(height):
        for x in range(width):
            # Convertir de BGR a HSV y obtener los valores HSV
            b, g, r = image[y, x]
            f = (192 - y)/117
            # saturation = saturation if f >= 1 else saturation * f

            # Aplicar fuzzificación y clasificación
            red_category = red_intensity(r,r_mode)
            green_category = green_intensity(g,g_mode)
            blue_category = blue_intensity(b,b_mode)
            classification = classify_pixel(red_category,blue_category,green_category)
            output_image[y, x] = colors[classification]  # Asignar color basado en la clasificación
            binary_image[y, x] = 1 if classification == "obstacle" else 0

    
    mask_matrix = np.zeros((8,16), dtype=np.uint8)
    cuadros = []
    for y in range(8):
        for x in range(16):
            mask_matrix[y,x] = np.count_nonzero(binary_image[y*(height//8):(y+1)*(height//8),x*(width//16):(x+1)*(width//16)])
            if mask_matrix[y,x] > 20:
                tiene_vecino = False
                for cuadro in cuadros:
                    if (y,x) in cuadro["vecinos"]:
                        nuevos_vecinos = {(y-1,x),(y+1,x),(y,x-1),(y,x+1)}
                        cuadro["vecinos"] = cuadro["vecinos"].union(nuevos_vecinos)
                        cuadro["x_init"] = min(cuadro["x_init"],x*(width//16))
                        cuadro["x_end"] = max(cuadro["x_end"],(x+1)*(width//16))
                        cuadro["y_init"] = min(cuadro["y_init"],y*(height//8))
                        cuadro["y_end"] = max(cuadro["y_end"],(y+1)*(height//8))
                        cuadro["centroid"] = (cuadro["x_init"]+cuadro["x_end"])//2,(cuadro["y_init"]+cuadro["y_end"])//2
                        cuadro["weight"] = cuadro["weight"] + mask_matrix[y,x]

                        tiene_vecino = True
                if not tiene_vecino:
                    cuadros.append({"vecinos":{(y-1,x),(y+1,x),(y,x-1),(y,x+1)},"x_init":x*(width//16),"x_end":(x+1)*(width//16),"y_init":y*(height//8),"y_end":(y+1)*(height//8),
                                    "centroid":((x*(width//16)+(x+1)*(width//16))//2,(y*(height//8)+(y+1)*(height//8))//2),"weight":mask_matrix[y,x]})

    return output_image,cuadros


def get_orientation(image:np.array):
    # Convertir la imagen a un formato binario basado en el color del obstáculo
    obstacle_color = np.array([224, 167, 41], dtype=np.uint8)
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
        output_path = f'/home/tille/Desktop/Tesis/WaSR-T/images/output_{i+1}.png'
        cv2.imwrite(output_path, classified_image)
        tics[f"image_{i+1}"].append(time.time()-tic1)
        # print(f"Imagen {i+1} procesada {j+1} veces.")

    # with open('times_fuzzy.pickle', 'wb') as f:
    #     pickle.dump(tics, f)