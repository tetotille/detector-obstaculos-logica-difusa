import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from os.path import dirname, abspath, join
from sys import argv
import matplotlib.pyplot as plt

# Crear variables difusas
P1 = ctrl.Antecedent(np.arange(0, 256, 1), 'P1')
P2 = ctrl.Antecedent(np.arange(0, 256, 1), 'P2')
P3 = ctrl.Antecedent(np.arange(0, 256, 1), 'P3')
P4 = ctrl.Antecedent(np.arange(0, 256, 1), 'P4')
P4_out = ctrl.Consequent(np.arange(0, 256, 1), 'P4_out')

    # Define the adjusted universe of discourse
universe = np.linspace(-800, 800, 100)

# Definir las funciones de pertenencia
for var in [P1, P2, P3, P4, P4_out]:
    var['low'] = fuzz.zmf(var.universe, 0, 255)
    var['medium'] = fuzz.gaussmf(var.universe, 127, 43)
    var['high'] = fuzz.smf(var.universe, 0, 255)


#Define the gaussmf function
def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2.) / (float(sigma) ** 2.))

# Define the membership functions
def define_mf(mean):
    return {
        'low': gaussmf(universe, 43, 43),
        'medium': gaussmf(universe, 127 , 43),
        'high': gaussmf(universe, 255 , 43)
    }

# Define means for P1, P2, P3
P1_mean = 0
P2_mean = 0
P3_mean = 0

# Calculate membership functions
P1_mfs = define_mf(P1_mean)
P2_mfs = define_mf(P2_mean)
P3_mfs = define_mf(P3_mean)

# Plotting the membership functions
plt.figure(figsize=(10, 6))

plt.plot(universe, P1_mfs['low'], label='P1 Low', linestyle='--')
plt.plot(universe, P1_mfs['medium'], label='P1 Medium', linestyle='-.')
plt.plot(universe, P1_mfs['high'], label='P1 High', linestyle='-')

plt.plot(universe, P2_mfs['low'], label='P2 Low', linestyle='--')
plt.plot(universe, P2_mfs['medium'], label='P2 Medium', linestyle='-.')
plt.plot(universe, P2_mfs['high'], label='P2 High', linestyle='-')

plt.plot(universe, P3_mfs['low'], label='P3 Low', linestyle='--')
plt.plot(universe, P3_mfs['medium'], label='P3 Medium', linestyle='-.')
plt.plot(universe, P3_mfs['high'], label='P3 High', linestyle='-')

plt.title('Adjusted Gaussian Membership Functions')
plt.xlabel('Value')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.show()

# Definir las reglas difusas
rule1 = ctrl.Rule(P1['low'] & P2['low'], P4_out['low'])
rule2 = ctrl.Rule(P1['medium'] & P2['medium'], P4_out['medium'])
rule3 = ctrl.Rule(P1['high'] & P2['high'], P4_out['high'])
rule4 = ctrl.Rule(P1['medium'] & P3['low'], P4_out['low'])
rule5 = ctrl.Rule(P2['medium'] & P3['low'], P4_out['low'])
rule6 = ctrl.Rule(P4['low'] & P2['medium'], P4_out['low'])
rule7 = ctrl.Rule(P4['low'] & P1['medium'], P4_out['low'])

# Crear el sistema de control difuso
edge_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
edge_detection = ctrl.ControlSystemSimulation(edge_ctrl)


def detectar_horizonte(image):
    """ Halla la línea del horizonte a través de las colisiones de dos gradientes de color.

    Args:
        image (Image): Imagen del agua

    Returns:
        image, transition_index: retorna la imagen dibujada con la línea del horizonte y la línea completa del horizonte
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = np.gradient(blurred_image, axis=0)
    smoothed_gradient = np.convolve(np.mean(gradient, axis=1), np.ones(15)/15, mode='same')

    transition_index = np.argmax(np.abs(smoothed_gradient))
    horizon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if transition_index is not None:
        cv2.line(image, (0, transition_index), (image.shape[1], transition_index), (0, 255, 0), thickness=2)
        horizon_mask[transition_index-2:transition_index+2, :] = 255  # Crear una máscara binaria
    cv2.imshow('horizontada', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image, horizon_mask

def segment_image(image, horizon_mask):
    rows, cols = image.shape[:2]
    horizon_line = np.argmax(horizon_mask, axis=0)  # Encuentra la línea del horizonte
    
    sea_mask = np.zeros((rows, cols), dtype=np.uint8)
    sky_mask = np.zeros((rows, cols), dtype=np.uint8)
    
    for col in range(cols):
        horizon_row = horizon_line[col]
        sea_mask[horizon_row:, col] = 255
        sky_mask[:horizon_row, col] = 255
    
    return sea_mask, horizon_mask, sky_mask


def detect_horizon(image_path):
    # Cargar la imagen
    print(f"Cargando imagen desde: {image_path}")  # Imprimir la ruta de la imagen para verificación
    image = cv2.imread(image_path)

    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        raise ValueError("La imagen no se pudo cargar. Verifica la ruta del archivo y asegúrate de que el archivo exista.")
    image=cv2.resize(image, (300, 300))

    # Detectar el horizonte
    horizonte_image, horizon_mask = detectar_horizonte(image)

    # Segmentar la imagen en mar, horizonte y cielo
    sea_mask, horizon_mask, sky_mask = segment_image(image, horizon_mask)

    # Detectar objetos en el mar usando la lógica difusa
    objetos_detectados = detectar_objetos_en_el_mar(image, sea_mask)

    # Mostrar la imagen con el horizonte detectado y los objetos
    resized_image = cv2.resize(image, (300, 300))  # Ajustar el tamaño para mejor visualización
    cv2.imshow('Horizon Detection and Object Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_objetos_en_el_mar(image, sea_mask):

    if image is None:
        print("Error: Image not loaded properly.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rezides=cv2.resize(blurred, (300,300))
        # Convertir la imagen a escala de grises
    
    # Aplicar la máscara del mar
    sea_region = cv2.bitwise_and(rezides, rezides, mask=sea_mask)
    cv2.imshow('mascara mar', sea_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crear una imagen para guardar los bordes difusos
    fuzzy_edge_image = np.zeros_like(sea_region, dtype=np.uint8)
    

    # Obtener las dimensiones de la imagen
    height, width = rezides.shape

    # Recorrer cada píxel de la imagen
    for y in range(0, height - 1):
        for x in range(0, width - 1):
            # Obtener los valores de los píxeles vecinos
            pixel_P1 = sea_region[y, x]
            pixel_P2 = sea_region[y, x + 1]
            pixel_P3 = sea_region[y + 1, x]
            pixel_P4 = sea_region[y + 1, x + 1]

            # Asignar los valores a las variables difusas
            edge_detection.input['P1'] = pixel_P1
            edge_detection.input['P2'] = pixel_P2
            edge_detection.input['P3'] = pixel_P3
            edge_detection.input['P4'] = pixel_P4

            # Calcular el valor de salida
            edge_detection.compute()
            fuzzy_edge_image[y + 1, x + 1] = edge_detection.output['P4_out']


        # Dibujar los contornos en la imagen original

            # Umbralizar la imagen resultante para obtener los contornos
    _, thresh = cv2.threshold(fuzzy_edge_image, 127, 255, cv2.THRESH_BINARY)

        # Encontrar los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Aplicar la máscara del mar
    print(fuzzy_edge_image)
    #contours = cv2.bitwise_and(rezides, rezides, mask=fuzzy_edge_image)
    cv2.imshow('contours', fuzzy_edge_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rezides2=cv2.resize(image, (300, 300))
    contour_image = rezides2.copy()
    #blended = cv2.addWeighted(contours, 0.5, contour_image, 0.5, 0)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    # Mostrar la imagen con los contornos detectados
    #cv2.imshow('contours', blended)
    cv2.imshow('contours2', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



if __name__ == "__main__":

    # Ruta a la imagen
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")

    detect_horizon(filename)

