import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definir las funciones de membresía para los píxeles vecinos y el píxel central
def define_membership_functions():
    x_pixel = np.arange(0, 1.01, 0.01)
    
    # Crear las variables difusas
    C1 = ctrl.Antecedent(x_pixel, 'C1')
    C2 = ctrl.Antecedent(x_pixel, 'C2')
    C3 = ctrl.Antecedent(x_pixel, 'C3')
    C4 = ctrl.Antecedent(x_pixel, 'C4')
    C5 = ctrl.Antecedent(x_pixel, 'C5')
    C6 = ctrl.Antecedent(x_pixel, 'C6')
    C7 = ctrl.Antecedent(x_pixel, 'C7')
    C8 = ctrl.Antecedent(x_pixel, 'C8')
    C9 = ctrl.Antecedent(x_pixel, 'C9')
    
    edge = ctrl.Consequent(np.arange(0, 2, 1), 'edge')
    
    # Definir funciones de membresía para cada vecino
    for C in [C1, C2, C3, C4, C5, C6, C7, C8, C9]:
        C['low'] = fuzz.trimf(C.universe, [0, 0, 0.5])
        C['high'] = fuzz.trimf(C.universe, [0.5, 1, 1])
    
    edge['no'] = fuzz.trimf(edge.universe, [0, 0, 1])
    edge['yes'] = fuzz.trimf(edge.universe, [1, 1, 1])
    
    return C1, C2, C3, C4, C5, C6, C7, C8, C9, edge

# Definir las reglas difusas
def define_rules(C1, C2, C3, C4, C5, C6, C7, C8, C9, edge):
    rule1 = ctrl.Rule(C1['high'] & C3['high'] & C5['high'] & 
                      C2['low'] & C4['low'] & C6['low'] & 
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule2 = ctrl.Rule(C5['high'] & C7['high'] & C9['high'] & 
                      C1['low'] & C2['low'] & C3['low'] & 
                      C4['low'] & C6['low'] & C8['low'], edge['yes'])
    
    # ... añadir las otras reglas aquí
    rule12 = ctrl.Rule(C1['high'] & C5['high'] & C9['high'] & 
                       C2['low'] & C3['low'] & C4['low'] & 
                       C6['low'] & C7['low'] & C8['low'], edge['yes'])
    
    rules = [rule1, rule2, rule12]
    return rules

# Crear el sistema de control difuso
def create_fuzzy_system():
    C1, C2, C3, C4, C5, C6, C7, C8, C9, edge = define_membership_functions()
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
            edge_detect.input['C2'] = neighbor_values[1]
            edge_detect.input['C3'] = neighbor_values[2]
            edge_detect.input['C4'] = neighbor_values[3]
            edge_detect.input['C5'] = neighbor_values[4]
            edge_detect.input['C6'] = neighbor_values[5]
            edge_detect.input['C7'] = neighbor_values[6]
            edge_detect.input['C8'] = neighbor_values[7]
            edge_detect.input['C9'] = neighbor_values[8]
            
            edge_detect.compute()
            
            edge_image[i, j] = 1 if edge_detect.output['edge'] >= 0.5 else 0
            
    return edge_image

# Función para cargar una imagen desde la computadora
def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

# Función principal para cargar imagen, aplicar detección de bordes difusa y otros análisis
def main(file_path):
    # Cargar imagen desde la computadora
    image = load_image(file_path)
    
    # Aplicar detección de bordes difusa
    fuzzy_image = image.astype(float) / 255.0  # Normalizar la imagen entre 0 y 1
    edge_detect = create_fuzzy_system()
    edge_image = apply_fuzzy_rules_to_image(fuzzy_image, edge_detect)
    
    # Mostrar resultados
    cv2.imshow('Original Image', image)
    cv2.imshow('Fuzzy Edge Detected Image', edge_image * 255)  # Escalar a 0-255 para visualizar
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejemplo de uso
if __name__ == "__main__":
    file_path = 'path/to/your/image.jpg'  # Ruta de la imagen en tu computadora
    main(file_path)
