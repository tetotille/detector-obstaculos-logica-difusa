import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from os.path import dirname, abspath, join
from sys import argv
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
#from hsv_filter import filter_h


# Definir las funciones de membresía para los píxeles vecinos y el píxel central
def define_membership_functions(image):
    min_pixel = np.min(image)
    max_pixel = np.max(image)
    universe = np.arange(min_pixel, max_pixel)/256  # Normalización entre 0 y 1
    # print(image[192, 1918], image[192, 1919], image[192, 1917], image[191, 1918], image[191, 1917], image[191, 1919], image[193, 1917], image[193, 1918], image[193, 1919])
    universe2= (np.arange(0, 256)/256)
    # Crear las variables difusas usando ctrl.Antecedent y ctrl.Consequent
    # Estas 9 variables indican los 9 pixeles que existen alrededor del elegido son las 9 variables de pixeles difusos
    C1 = ctrl.Antecedent(universe, 'C1')
    C2 = ctrl.Antecedent(universe, 'C2')
    C3 = ctrl.Antecedent(universe, 'C3')
    C4 = ctrl.Antecedent(universe, 'C4')
    C5 = ctrl.Antecedent(universe, 'C5')
    C6 = ctrl.Antecedent(universe, 'C6')
    C7 = ctrl.Antecedent(universe, 'C7')
    C8 = ctrl.Antecedent(universe, 'C8')
    C9 = ctrl.Antecedent(universe, 'C9')

    # Esto sería la inicialización de las funciones de pertenencia
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
        C['low'] = fuzz.trimf(universe, [0, 0, 1])
        C['high'] = fuzz.trimf(universe, [0, 1, 1])
    
    edge['low'] = fuzz.trimf(universe2, [0, 0, 0.5])
    edge['high']= fuzz.trimf(universe2, [0.5, 1, 1])
    edge['yes'] = fuzz.trimf(universe2, [0, 0.5, 1])
    
        # Definir el universo de discurso
    universe = np.arange(0, 1.01, 0.01)
    universe2 = np.arange(0, 1.01, 0.01)

    ##### Se hace una copia exclusivamente para dibujar y ver las funciones de membresía #####
    # Definir las funciones de membresía
    C_low = fuzz.trimf(universe, [0, 0, 1])
    C_high = fuzz.trimf(universe, [0, 1, 1])

    edge_low = fuzz.trimf(universe2, [0, 0, 0.5])
    edge_high= fuzz.trimf(universe2, [0.5, 1, 1])
    edge_yes = fuzz.trimf(universe2, [0, 0.5, 1])

    # Graficar las funciones de membresía
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Funciones de membresía para C (low y high)
    ax[0].plot(universe, C_low, 'b', linewidth=1.5, label='Low')
    ax[0].plot(universe, C_high, 'r', linewidth=1.5, label='High')
    ax[0].set_title('Funciones de Membresía para C')
    ax[0].legend()

    # Funciones de membresía para edge (no y yes)
    ax[1].plot(universe2, edge_low, 'b', linewidth=1.5, label='No')
    ax[1].plot(universe2, edge_yes, 'r', linewidth=1.5, label='Yes')
    ax[1].plot(universe2, edge_high, 'y', linewidth=1.5, label='Yes')   
    ax[1].set_title('Funciones de Membresía para Edge')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
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
                       C6['low'] & C7['low'] & C8['low'], edge['low'])
    
    rule14 = ctrl.Rule(C5['high'] & C3['high'] & C1['high'] & 
                      C9['high'] & C2['high'] & C7['high'] & 
                      C4['high'] & C6['high'] & C8['high'], edge['high'])
    
    rule15 = ctrl.Rule(C5['low'] & C3['low'] & C1['low'] & 
                      C2['low'] & C4['low'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule16= ctrl.Rule(C5['low'] & C3['low'] & C6['low'] & 
                      C9['low'] & C8['low'] & C2['low'] & 
                      C1['high'] & C4['high'] & C7['high'], edge['yes']) 
    
    rule17= ctrl.Rule(C5['low'] & C4['low'] & C6['low'] & 
                      C9['low'] & C8['low'] & C7['low'] & 
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
    
    rule27 = ctrl.Rule(C1['low'] & C5['high'] & C4['high'] & 
                     C2['low'] & C6['high'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['high'])
    
    rule28= ctrl.Rule(C1['high'] & C5['low'] & C4['low'] & 
                     C2['high'] & C6['low'] & C3['high'] & 
                     C8['high'] & C9['high'] & C7['high'], edge['low'])
    
    rule29= ctrl.Rule(C1['low'] & C5['high'] & C4['low'] & 
                     C2['high'] & C6['low'] & C3['low'] & 
                     C8['high'] & C9['low'] & C7['low'], edge['high'])
    
    rule30= ctrl.Rule(C1['high'] & C5['low'] & C4['high'] & 
                     C2['low'] & C6['high'] & C3['high'] & 
                     C8['low'] & C9['high'] & C7['high'], edge['low'])   
    
    rule31 = ctrl.Rule(
    (C1['high'] & C2['high'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C1['high'] & C3['high'] & C2['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C1['high'] & C4['high'] & C2['low'] & C3['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C1['high'] & C5['high'] & C2['low'] & C3['low'] & C4['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C1['high'] & C6['high'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C1['high'] & C7['high'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C8['low'] & C9['low']) |
     (C1['high'] & C8['high'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C9['low']) |
     (C1['high'] & C9['high'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low']) |
     (C2['high'] & C3['high'] & C1['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C2['high'] & C4['high'] & C1['low'] & C3['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C2['high'] & C5['high'] & C1['low'] & C3['low'] & C4['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C2['high'] & C6['high'] & C1['low'] & C3['low'] & C4['low'] & C5['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C2['high'] & C7['high'] & C1['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C8['low'] & C9['low']) |
     (C2['high'] & C8['high'] & C1['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C9['low']) |
     (C2['high'] & C9['high'] & C1['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low']) |
     (C3['high'] & C4['high'] & C1['low'] & C2['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C3['high'] & C5['high'] & C1['low'] & C2['low'] & C4['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C3['high'] & C6['high'] & C1['low'] & C2['low'] & C4['low'] & C5['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C3['high'] & C7['high'] & C1['low'] & C2['low'] & C4['low'] & C5['low'] & C6['low'] & C8['low'] & C9['low']) |
     (C3['high'] & C8['high'] & C1['low'] & C2['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C9['low']) |
     (C3['high'] & C9['high'] & C1['low'] & C2['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low']) |
     (C4['high'] & C5['high'] & C1['low'] & C2['low'] & C3['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C4['high'] & C6['high'] & C1['low'] & C2['low'] & C3['low'] & C5['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C4['high'] & C7['high'] & C1['low'] & C2['low'] & C3['low'] & C5['low'] & C6['low'] & C8['low'] & C9['low']) |
     (C4['high'] & C8['high'] & C1['low'] & C2['low'] & C3['low'] & C5['low'] & C6['low'] & C7['low'] & C9['low']) |
     (C4['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low']) |
     (C5['high'] & C6['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C7['low'] & C8['low'] & C9['low']) |
     (C5['high'] & C7['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C6['low'] & C8['low'] & C9['low']) |
     (C5['high'] & C8['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C6['low'] & C7['low'] & C9['low']) |
     (C5['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C6['low'] & C7['low'] & C8['low']) |
     (C6['high'] & C7['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C8['low'] & C9['low']) |
     (C6['high'] & C8['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C7['low'] & C9['low']) |
     (C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C7['low'] & C8['low']) |
     (C7['high'] & C8['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C9['low']) |
     (C7['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C8['low']) |
     (C8['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low']),
    edge['low'])

    rule32 = ctrl.Rule(
    (C1['low'] & C2['low'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C1['low'] & C3['low'] & C2['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C1['low'] & C4['low'] & C2['high'] & C3['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) | # type: ignore
     (C1['low'] & C5['low'] & C2['high'] & C3['high'] & C4['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C1['low'] & C6['low'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C1['low'] & C7['low'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C8['high'] & C9['high']) |
     (C1['low'] & C8['low'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C9['high']) |
     (C1['low'] & C9['low'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high']) |
     (C2['low'] & C3['low'] & C1['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C2['low'] & C4['low'] & C1['high'] & C3['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C2['low'] & C5['low'] & C1['high'] & C3['high'] & C4['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C2['low'] & C6['low'] & C1['high'] & C3['high'] & C4['high'] & C5['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C2['low'] & C7['low'] & C1['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C8['high'] & C9['high']) |
     (C2['low'] & C8['low'] & C1['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C9['high']) |
     (C2['low'] & C9['low'] & C1['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high']) |
     (C3['low'] & C4['low'] & C1['high'] & C2['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C3['low'] & C5['low'] & C1['high'] & C2['high'] & C4['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C3['low'] & C6['low'] & C1['high'] & C2['high'] & C4['high'] & C5['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C3['low'] & C7['low'] & C1['high'] & C2['high'] & C4['high'] & C5['high'] & C6['high'] & C8['high'] & C9['high']) |
     (C3['low'] & C8['low'] & C1['high'] & C2['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C9['high']) |
     (C3['low'] & C9['low'] & C1['high'] & C2['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high']) |
     (C4['low'] & C5['low'] & C1['high'] & C2['high'] & C3['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C4['low'] & C6['low'] & C1['high'] & C2['high'] & C3['high'] & C5['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C4['low'] & C7['low'] & C1['high'] & C2['high'] & C3['high'] & C5['high'] & C6['high'] & C8['high'] & C9['high']) |
     (C4['low'] & C8['low'] & C1['high'] & C2['high'] & C3['high'] & C5['high'] & C6['high'] & C7['high'] & C9['high']) |
     (C4['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high']) |
     (C5['low'] & C6['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C7['high'] & C8['high'] & C9['high']) |
     (C5['low'] & C7['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C6['high'] & C8['high'] & C9['high']) |
     (C5['low'] & C8['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C6['high'] & C7['high'] & C9['high']) |
     (C5['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C6['high'] & C7['high'] & C8['high']) |
     (C6['low'] & C7['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C8['high'] & C9['high']) |
     (C6['low'] & C8['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C7['high'] & C9['high']) |
     (C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C7['high'] & C8['high']) |
     (C7['low'] & C8['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C9['high']) |
     (C7['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C8['high']) |
     (C8['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high']),
    edge['high'])

    # Definir la regla para cuando solo una variable es 'low' y el resto son 'high'
    rule33 = ctrl.Rule((C1['low'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['low'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['low'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['high'] & C4['low'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['low'] & C6['high'] & C7['high'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['low'] & C7['high'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['low'] & C8['high'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['low'] & C9['high']) |
        (C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C6['high'] & C7['high'] & C8['high'] & C9['low']), edge['high'])

# Definir la regla para cuando solo una variable es 'high' y el resto son 'low'
    rule34 = ctrl.Rule((C1['high'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['high'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['high'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['low'] & C4['high'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['high'] & C6['low'] & C7['low'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['high'] & C7['low'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['high'] & C8['low'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['high'] & C9['low']) |
        (C1['low'] & C2['low'] & C3['low'] & C4['low'] & C5['low'] & C6['low'] & C7['low'] & C8['low'] & C9['high']), edge['low'])
    
    rule35 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['high'])

    rule36 = ctrl.Rule(C1['high'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['high'])

    rule37 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['low'], edge['low'])

    rule38 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['low'])

    rule39 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['low'])
    
    rule40 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])

    rule41 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['high'], edge['high'])

    rule42 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['low'], edge['high'])

    rule43 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['high'], edge['high'])

    rule44 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['high'], edge['low'])
    
    rule45 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['low'], edge['low'])
    
    rule46 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['high'], edge['high'])
    
    rule46 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['high'])
    
    rule47 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['low'])
    
    rule48 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] & C4['low'] & C5['high'] & C6['high'] & C7['low'] & C8['low'] & C9['low'], edge['high'])
    rule49 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] & C4['low'] & C5['low'] & C6['low'] & C7['high'] & C8['high'] & C9['low'], edge['low'])
    rule50 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] & C4['high'] & C5['low'] & C6['high'] & C7['low'] & C8['low'] & C9['high'], edge['low'])
    rule51 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] & C4['high'] & C5['high'] & C6['low'] & C7['low'] & C8['low'] & C9['high'], edge['high'])
    rule52 = ctrl.Rule(C1['low'] & C2['high'] & C3['low'] & C4['low'] & C5['high'] & C6['high'] & C7['low'] & C8['low'] & C9['high'], edge['high'])

# Puedes continuar añadiendo más reglas según sea necesario


    
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7,rule8, rule9, rule10, rule11, rule12,
        rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, 
        rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36,
        rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48,
        rule49, rule50, rule51, rule52]
    return rules

# Crear el sistema de control difuso
def create_fuzzy_system(image):
    C1, C2, C3, C4, C5, C6, C7, C8, C9, edge = define_membership_functions(image)
    rules = define_rules(C1, C2, C3, C4, C5, C6, C7, C8, C9, edge)
    
    edge_ctrl = ctrl.ControlSystem(rules)
    edge_detect = ctrl.ControlSystemSimulation(edge_ctrl)
    
    return edge_detect

# Aplicar las reglas difusas a la imagen
def apply_fuzzy_rules_to_image(fuzzy_image, edge_detect):
    rows, cols = fuzzy_image.shape
    edge_image = np.zeros((rows, cols), dtype=float)
    
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
            #print(j)
            edge_detect.input['C2'] = neighbor_values[1]
            edge_detect.input['C3'] = neighbor_values[2]
            edge_detect.input['C4'] = neighbor_values[3]
            edge_detect.input['C5'] = neighbor_values[4]
            edge_detect.input['C6'] = neighbor_values[5]
            edge_detect.input['C7'] = neighbor_values[6]
            edge_detect.input['C8'] = neighbor_values[7]
            edge_detect.input['C9'] = neighbor_values[8]
            #print(neighbor_values[8])

            try:
                edge_detect.compute()          
            except AssertionError as e:
                print(f"Error en la posición ({i}, {j}): {e}")
                edge_image[i, j] =  fuzzy_image[i, j]
                continue

            edge_image[i,j] = edge_detect.output["edge"]

        # if edge_detect.output['edge'] == 0.5:  # Si es 'yes'
        #     edge_image[i, j] = 0.5 
        # elif edge_detect.output['edge'] > 0.5: 
        #     edge_image[i, j] = 1
        # elif edge_detect.output['edge'] < 0.5:
        #     edge_image[i, j] = 0
                  
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
        filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
    image=cv2.imread(filename)
    x,y,a = image.shape
    imagen=cv2.resize(image, (300, int(x*300/y)))
    gray=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    #image2=filter_h(filename)
    cv2.imshow('Original Image', gray)
    cv2.waitKey(0)

    # Aplicar detección de bordes difusa
    fuzzy_image = gray.astype(float) / 256.00000000  # Normalizar la imagen entre 0 y 1
    edge_detect = create_fuzzy_system (gray)
    edge_image = apply_fuzzy_rules_to_image (fuzzy_image, edge_detect)
    #print(edge_image)
    edge_image_uint8 = (edge_image * 255).astype(np.uint8)
    print(edge_image_uint8)
    # Mostrar resultados
    # cv2.imshow('Original Image', image)
    cv2.imshow('Fuzzy Edge Detected Image', edge_image_uint8)  # Escalar a 0-255 para visualizar
    cv2.waitKey(0)
    cv2.imwrite("test.jpeg",edge_image_uint8)
    cv2.destroyAllWindows()

# Ruta a la imagen

main()