import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from os.path import dirname, abspath, join
import sys
import matplotlib.pyplot as plt

# Obtener la ruta de la imagen
argv = sys.argv
if len(argv)>1:
    filename = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))),"img/barco.jpg")

# Cargar la imagen y convertirla a escala de grises
image = cv2.imread(filename)

if image is None:
    print("Error: Image not loaded properly.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rezides=cv2.resize(gray, (300,300))

# Crear variables difusas
P1 = ctrl.Antecedent(np.arange(0, 256, 1), 'P1')
P2 = ctrl.Antecedent(np.arange(0, 256, 1), 'P2')
P3 = ctrl.Antecedent(np.arange(0, 256, 1), 'P3')
P4 = ctrl.Antecedent(np.arange(0, 256, 1), 'P4')
P4_out = ctrl.Consequent(np.arange(0, 256, 1), 'P4_out')

    # Define the adjusted universe of discourse
#universe = np.linspace(0, 255, 1)

# Definir las funciones de pertenencia
for var in [P1, P2, P3, P4, P4_out]:
    var['low'] = fuzz.gaussmf(var.universe, 0,43)
    var['medium'] = fuzz.gaussmf(var.universe, 127,43)
    var['high'] = fuzz.gaussmf(var.universe, 255, 43)


# Define the gaussmf function
#def gaussmf(x, mean, sigma):
#    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# Define the membership functions
#def define_mf(mean):
#    return {
#        'low': gaussmf(universe, mean, 43),
#        'medium': gaussmf(universe, mean +127 , 43),
#        'high': gaussmf(universe, mean +400 , 127)
#    }

# Define means for P1, P2, P3
#P1_mean = 0
#P2_mean = 127
#P3_mean = 400

# Calculate membership functions
#P1_mfs = define_mf(P1_mean)
#P2_mfs = define_mf(P2_mean)
#P3_mfs = define_mf(P3_mean)

# Plotting the membership functions
#plt.figure(figsize=(10, 6))

#plt.plot(universe, P1_mfs['low'], label='P1 Low', linestyle='--')
#plt.plot(universe, P1_mfs['medium'], label='P1 Medium', linestyle='-.')
#plt.plot(universe, P1_mfs['high'], label='P1 High', linestyle='-')

#plt.plot(universe, P2_mfs['low'], label='P2 Low', linestyle='--')
#plt.plot(universe, P2_mfs['medium'], label='P2 Medium', linestyle='-.')
#plt.plot(universe, P2_mfs['high'], label='P2 High', linestyle='-')

#plt.plot(universe, P3_mfs['low'], label='P3 Low', linestyle='--')
#plt.plot(universe, P3_mfs['medium'], label='P3 Medium', linestyle='-.')
#plt.plot(universe, P3_mfs['high'], label='P3 High', linestyle='-')

#plt.title('Adjusted Gaussian Membership Functions')
#plt.xlabel('Value')
#plt.ylabel('Membership')
#plt.legend()
#plt.grid(True)
#plt.show()

# Definir las reglas difusas
rule1 = ctrl.Rule(P1['low'] & P2['low'], P4_out['low'])
rule2 = ctrl.Rule(P1['medium'] & P2['medium'], P4_out['high'])
rule3 = ctrl.Rule(P1['high'] & P2['high'], P4_out['high'])
rule4 = ctrl.Rule(P1['medium'] & P3['low'], P4_out['high'])
rule5 = ctrl.Rule(P2['medium'] & P3['low'], P4_out['high'])
rule6 = ctrl.Rule(P4['low'] & P2['medium'], P4_out['low'])
rule7 = ctrl.Rule(P4['low'] & P1['medium'], P4_out['low'])

# Crear el sistema de control difuso
edge_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
edge_detection = ctrl.ControlSystemSimulation(edge_ctrl)

# Crear una imagen para guardar los bordes difusos
fuzzy_edge_image = np.zeros_like(rezides, dtype=np.uint8)

# Obtener las dimensiones de la imagen
height, width = rezides.shape

# Recorrer cada píxel de la imagen
for y in range(0, height - 1):
    for x in range(0, width - 1):
        # Obtener los valores de los píxeles vecinos
        pixel_P1 = rezides[y, x]
        pixel_P2 = rezides[y, x + 1]
        pixel_P3 = rezides[y + 1, x]
        pixel_P4 = rezides[y + 1, x + 1]

        # Asignar los valores a las variables difusas
        edge_detection.input['P1'] = pixel_P1
        edge_detection.input['P2'] = pixel_P2
        edge_detection.input['P3'] = pixel_P3
        edge_detection.input['P4'] = pixel_P4

        # Calcular el valor de salida
        edge_detection.compute()
        fuzzy_edge_image[y + 1, x + 1] = edge_detection.output['P4_out']

# Umbralizar la imagen resultante para obtener los contornos
_, thresh = cv2.threshold(fuzzy_edge_image, 100, 127, cv2.THRESH_BINARY)

# Encontrar los contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
rezides2=cv2.resize(image, (300, 300))
contour_image = rezides2.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos detectados
cv2.imshow('Contornos Detectados', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
