import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from os.path import dirname, abspath, join
from sys import argv
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from hsv_filter import filter_h


# Definir las funciones de membresía para los píxeles vecinos y el píxel central
def define_membership_functions(filename):
    # Leer la imagen
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image2=filter_h(filename)
    imagen=cv2.resize(image, (300, 300))  # Lee la imagen en escala de grises
    #print(imagen)
    #print(np.max(imagen))
    x_pixel = imagen.flatten()  # Obtener todos los píxeles de la imagen como un array 1D
    min_pixel = np.min(image2)
    max_pixel = np.max(image2)
    universe = (np.arange(min_pixel, max_pixel ) / 256)  # Normalización entre 0 y 1
    universe2= (np.arange(0, 256)/256)
    # Crear las variables difusas usando ctrl.Antecedent y ctrl.Consequent
    C1 = ctrl.Antecedent(universe, 'C1')
    C2 = ctrl.Antecedent(universe, 'C2')
    C3 = ctrl.Antecedent(universe, 'C3')
    C4 = ctrl.Antecedent(universe, 'C4')
    C5 = ctrl.Antecedent(universe, 'C5')
    C6 = ctrl.Antecedent(universe, 'C6')
    C7 = ctrl.Antecedent(universe, 'C7')
    C8 = ctrl.Antecedent(universe, 'C8')
    C9 = ctrl.Antecedent(universe, 'C9')

    for C in [C1, C2, C3, C4, C5, C6, C7, C8, C9]:
        C.automf(2, names=['low', 'high']) # Dividir en 2 funciones de membresía

    # Definir el consecuente (edge) con un rango adecuado
    edge = ctrl.Consequent(universe2, 'edge')

     # Visualización opcional de las funciones de membresía
    #fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    #for i, var in enumerate([C1, C2, C3, C4, C5, C6, C7, C8, C9]):
        #var.view(ax=axs[i//3, i%3])

    #edge.view()

    #plt.show()

    # Definir funciones de membresía para cada vecino
    for C in [C1, C2, C3, C4, C5, C6, C7, C8, C9]:
        C['low'] = fuzz.trimf(universe, [0, 0, 0.5])
        C['high'] = fuzz.trimf(universe, [0.5, 1, 1])
    
    edge['no'] = fuzz.trimf(universe2, [0, 0, 1])
    edge['yes'] = fuzz.trimf(universe2, [0, 1, 1])
    
    return C1, C2, C3, C4, C5, C6, C7, C8, C9, edge

# Definir las reglas difusas
def define_rules(C1, C2, C3, C4, C5, C6, C7, C8, C9, edge):
    rule1 = ctrl.Rule(C1['high'] & C3['high'] & C5['high'] & 
                      C2['high'] & C4['high'] & C6['high'] & 
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule2 = ctrl.Rule(C5['high'] & C7['high'] & C9['high'] & 
                      C1['low'] & C2['low'] & C3['low'] & 
                      C4['high'] & C6['high'] & C8['high'], edge['yes'])
    
    rule3 = ctrl.Rule(C5['high'] & C3['low'] & C1['low'] & 
                      C9['high'] & C2['high'] & C7['low'] & 
                      C4['high'] & C6['high'] & C8['high'], edge['yes'])
    
    rule4 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] & 
                      C9['low'] & C6['low'] & C3['low'] & 
                      C5['high'] & C7['high'] & C8['high'], edge['yes'])
    
    rule5 = ctrl.Rule(C1['low'] & C2['low'] & C4['low'] & 
                      C5['low'] & C6['high'] & C3['high'] & 
                      C7['high'] & C9['high'] & C8['high'], edge['yes'])
    
    rule6 = ctrl.Rule(C2['high'] & C1['high'] & C3['high'] & 
                      C8['low'] & C6['high'] & C9['high'] & 
                      C4['low'] & C5['low'] & C7['low'], edge['yes'])
    
    rule7 = ctrl.Rule(C4['high'] & C7['high'] & C8['high'] & 
                     C1['high'] & C6['low'] & C3['low'] & 
                     C2['low'] & C9['high'] & C5['low'], edge['yes'])
    
    rule8 = ctrl.Rule(C6['low'] & C5['low'] & C8['low'] & 
                     C1['high'] & C4['high'] & C3['high'] & 
                     C2['high'] & C9['low'] & C7['high'], edge['yes'])
    
    rule9 = ctrl.Rule(C4['low'] & C5['low'] & C1['low'] & 
                     C6['high'] & C8['high'] & C9['high'] & 
                     C2['low'] & C3['low'] & C7['low'], edge['yes'])
    
    rule10 = ctrl.Rule(C3['low'] & C5['low'] & C9['low'] & 
                     C1['low'] & C6['low'] & C3['low'] & 
                     C4['high'] & C8['high'] & C7['high'], edge['yes'])
    
    rule11 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] & 
                     C5['low'] & C6['low'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])
    
    # ... añadir las otras reglas aquí

    rule12 = ctrl.Rule(C1['low'] & C5['low'] & C9['low'] & 
                       C2['high'] & C3['high'] & C4['low'] & 
                       C6['high'] & C7['low'] & C8['low'], edge['yes'])
    
    rule13 =ctrl.Rule(C1['low'] & C5['low'] & C9['low'] & 
                       C2['low'] & C3['low'] & C4['low'] & 
                       C6['low'] & C7['low'] & C8['low'], edge['no'])
    
    rule14 = ctrl.Rule(C5['high'] & C3['high'] & C1['high'] & 
                      C9['high'] & C2['high'] & C7['high'] & 
                      C4['high'] & C6['high'] & C8['high'], edge['no'])
    
    rule15 = ctrl.Rule(C5['low'] & C3['low'] & C1['low'] & 
                      C2['low'] & C4['low'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule16= ctrl.Rule(C5['low'] & C3['low'] & C6['low'] & 
                      C9['low'] & C8['low'] & C2['low'] & 
                      C1['high'] & C4['high'] & C7['high'], edge['yes']) 
    
    rule17= ctrl.Rule(C5['low'] & C4['low'] & C6['low'] & 
                      C9['low'] & C8['low'] & C2['low'] & 
                      C1['high'] & C2['high'] & C3['high'], edge['yes']) 
    
    rule18 = ctrl.Rule(C9['high'] & C3['high'] & C6['high'] & 
                      C7['low'] & C8['low'] & C5['low'] & 
                      C1['low'] & C2['low'] & C3['low'], edge['yes'])  
    
    #otro tipo de division
  
    rule19 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] & 
                      C5['high'] & C6['low'] & C3['low'] & 
                      C7['low'] & C9['low'] & C8['low'], edge['yes'])
    
    rule20 = ctrl.Rule(C2['low'] & C1['low'] & C3['low'] & 
                      C8['high'] & C6['low'] & C9['low'] & 
                      C4['high'] & C5['high'] & C7['high'], edge['yes']) 
    
    rule21 = ctrl.Rule(C4['low'] & C7['low'] & C8['low'] & 
                     C1['low'] & C6['high'] & C3['high'] & 
                     C2['high'] & C9['low'] & C5['high'], edge['yes'])
    
    rule22 = ctrl.Rule(C6['high'] & C5['high'] & C8['high'] & 
                     C1['low'] & C4['low'] & C3['low'] & 
                     C2['low'] & C9['high'] & C7['low'], edge['yes'])
    #Aca otro
    rule23 = ctrl.Rule(C4['high'] & C5['high'] & C1['high'] & 
                     C6['low'] & C8['low'] & C9['low'] & 
                     C2['high'] & C3['high'] & C7['high'], edge['yes'])
    
    rule24 = ctrl.Rule(C5['high'] & C6['high'] & C1['high'] & 
                     C4['low'] & C8['low'] & C7['low'] & 
                     C2['high'] & C3['high'] & C9['high'], edge['yes'])
    
    rule25 = ctrl.Rule(C3['high'] & C5['high'] & C9['high'] & 
                     C1['low'] & C2['low'] & C4['low'] & 
                     C6['high'] & C8['high'] & C7['high'], edge['yes'])
    
    rule26 = ctrl.Rule(C1['high'] & C5['high'] & C4['high'] & 
                     C2['low'] & C6['low'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])
    
    
    

    
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, ]
    return rules

# Crear el sistema de control difuso
def create_fuzzy_system(filename):
    C1, C2, C3, C4, C5, C6, C7, C8, C9, edge = define_membership_functions(filename)
    rules = define_rules(C1, C2, C3, C4, C5, C6, C7, C8, C9, edge)
    
    edge_ctrl = ctrl.ControlSystem(rules)
    edge_detect = ctrl.ControlSystemSimulation(edge_ctrl)
    
    return edge_detect

# Aplicar las reglas difusas a la imagen
def apply_fuzzy_rules_to_image(fuzzy_image, edge_detect):
    rows, cols = fuzzy_image.shape
    edge_image = np.zeros((rows, cols), dtype=int)
    
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1), 
        (0, -1),  (0, 0),  (0, 1),  
        (1, -1),  (1, 0),  (1, 1)   
    ]
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighbor_values = [fuzzy_image[i+di, j+dj] for di, dj in neighbors]
            
            edge_detect.input['C1'] = neighbor_values[0]
            print(i)
            edge_detect.input['C2'] = neighbor_values[1]
            edge_detect.input['C3'] = neighbor_values[2]
            edge_detect.input['C4'] = neighbor_values[3]
            edge_detect.input['C5'] = neighbor_values[4]
            edge_detect.input['C6'] = neighbor_values[5]
            edge_detect.input['C7'] = neighbor_values[6]
            edge_detect.input['C8'] = neighbor_values[7]
            edge_detect.input['C9'] = neighbor_values[8]
            print(neighbor_values[8])
            edge_detect.compute()
            
            edge_image[i, j] = 1 if edge_detect.output['edge'] >= 0.5 else 0
            #edge_image[i, j] = edge_detect.output['edge']
            
    return edge_image

# Función para cargar una imagen desde la computadora
def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

# Función principal para cargar imagen, aplicar detección de bordes difusa y otros análisis
def main():
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/Black_and_White.jpg")
    image=cv2.imread(filename)
    imagen=cv2.resize(image, (300, 300))
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2=filter_h(filename)


    # Aplicar detección de bordes difusa
    fuzzy_image = gray.astype(float) / 256.00  # Normalizar la imagen entre 0 y 1
    edge_detect = create_fuzzy_system (filename)
    edge_image = apply_fuzzy_rules_to_image (fuzzy_image, edge_detect)
    #print(edge_image)
    edge_image_uint8 = (edge_image * 255).astype(np.uint8)
    # Mostrar resultados
    cv2.imshow('Original Image', image2)
    cv2.imshow('Fuzzy Edge Detected Image', edge_image_uint8)  # Escalar a 0-255 para visualizar
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta a la imagen

main()