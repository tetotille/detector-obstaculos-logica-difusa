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
def define_membership_functions(image):
    x,y = image.shape
    imagen=cv2.resize(image, (200, int(x*200/y)))
    #gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_pixel = np.min(imagen)
    max_pixel = np.max(imagen)
    universe = np.arange(min_pixel, max_pixel)/256  # Normalización entre 0 y 1
    #print(gray[2, 100], gray[2, 99], gray[2, 101], gray[1, 99], gray[1, 100], gray[1, 101], gray[3, 99], gray[3, 100], gray[3,101])
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

    # Definir funciones de membresía para cada vecino
    for C in [C1, C2, C3, C4, C5, C6, C7, C8, C9]:
        C['low'] = fuzz.trimf(universe, [0, 0, 0.5])
        C['high'] = fuzz.trimf(universe, [0.49609375, 1, 1])
    
    edge['low'] = fuzz.trimf(universe2, [0, 0, 0.5])
    edge['high']= fuzz.trimf(universe2, [0.5, 1, 1])
    edge['yes'] = fuzz.trimf(universe2, [0.5, 0.5, 0.5])
    
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
    
    rule3 = ctrl.Rule(C5['high'] & C4['low'] & C1['low'] & 
                      C9['high'] & C2['high'] & C7['low'] & 
                      C3['high'] & C6['high'] & C8['high'], edge['yes'])
    
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
    
    rule28 = ctrl.Rule(C1['low'] & C5['low'] & C4['high'] & 
                     C2['low'] & C6['low'] & C3['low'] & 
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
        (C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & C5['high'] & C8['high'] & C7['high']) |
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
                   C7['high'] & C8['low'] & C9['high'], edge['low'])

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
    rule53 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule54 = ctrl.Rule(C1['high'] & C2['low'] & C3['high'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['high'] & C8['low'] & C9['high'], edge['yes'])
    rule55 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    rule56 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    rule57 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule58 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    rule59 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    rule60 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule61 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    rule62 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    rule63 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule64 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['low'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule65 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    rule66 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    rule67 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule68 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    rule69 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['low'] & C5['low'] & C6['low'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule70 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule71 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    rule72 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    rule73 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    rule74 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    rule75 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    rule76 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['high'] & C5['high'] & C6['high'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    rule77 =  ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low']) 
    
    rule78 =  ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high']) 
    
    rule79 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low']) 
    
    rule80 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes']) 
    rule81 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes']) 
    rule82 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule83 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])  
    rule84 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low']) 
    rule85 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high']) 
    rule86 =  ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes']) 
    rule87 =  ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low']) 
    rule88 =  ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes']) 
    rule89 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low']) 
    rule90 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule91 =  ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes']) 
    rule92 =  ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule93 =  ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule94 =  ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule95 =  ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule96 =  ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule97 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule98 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule99 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule100 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule101 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule102 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule103 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule104 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule105 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule106 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule107 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule108 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule109 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule110 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule111 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule112 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule113  = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule114 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule115 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule116 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule117 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule118 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule119 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule120 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule121 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule122 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule123 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule124 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule125 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule126 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule127 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule128 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule129 =  ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule130  = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule131 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule132 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule133 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule134 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule135 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule136 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule137  = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule138 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule139 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule140 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule141 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low'])
    rule142 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule143 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule144 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule145 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule146 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule147  = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule148 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule149 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule150 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['low'])
    rule151 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule152 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule153 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule154 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule155 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule156 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule157 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['low'])
    rule158 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule159 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule160 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule161 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule162 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule163 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule164 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule165 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule166 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule167 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule168 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule169 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule170 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule171 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule172 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule173 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule174 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule175 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule176 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule177 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule178 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule179 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low'])
    rule180 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule181 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule182 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule183 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule184 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule185 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
# Puedes continuar añadiendo más reglas según sea necesario
    rule186 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule187 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule188 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['low'])
    rule189 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule190 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule191 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule192 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule193 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule194 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['low'])
    rule195 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule196 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule197 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule198 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule199 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule200 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule201 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule202 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule203 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule204 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule205 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule206 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule207 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule208 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule209 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule210 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule211 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule212 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule213 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule214 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule215 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule216 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule217 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule218 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low'])
    rule219 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule220 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule221 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule222 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule223 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule224 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule225 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule226 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule227 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule228 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule229 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule230 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule231 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule232 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule233 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule234 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule235 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule236 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule237 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule238 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule239 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule240 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule241 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule242 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule243 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule244 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule245 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule246 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule247 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule248 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule249 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule250 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule251 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule252 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule253 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule254 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule255 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule256 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule257 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule258 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule259 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule260 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])    
    rule261 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] &
               C5['high'] & C7['low'] & C8['high'], edge['high'])  
    rule262 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])   
    rule263 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule264 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule265 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule266 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule267 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule268 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule269 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule270 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule271 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule272 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule273 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule274 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule275 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule276 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule277 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule278 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low'])
    rule279 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule280 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule281 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule282 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['high'])
    rule283 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['low'])
    rule284 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule285 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule286 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule287 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule288 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule289 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule290 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule291 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule292 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low'])
    rule293 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule294 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule295 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule296 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule297 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule298 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule299 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule300 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule301 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule302 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule303 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule304 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule305 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule306 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule307 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule308 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule309 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule310 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule311 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule312 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule313 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule314 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['low'])
    rule315 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule316 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule317 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule318 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule319 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule320 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule321 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule322 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule323 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule324 = ctrl.Rule(C6['low'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['high'])
    rule325 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule326 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule327 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['yes'])
    rule328 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule329 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['yes'])
    rule330 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule331 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule332 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['high'])
    rule333 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['low'], edge['high'])
    rule334 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['low'])
    rule335 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule336 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule337 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule338 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule339 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['high'] & C8['low'], edge['low'])
    rule340 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['yes'])
    rule341 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule342 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['high'])
    rule343 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule344 = ctrl.Rule(C6['high'] & C9['high'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['low'], edge['yes'])
    rule345 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule346 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['high'], edge['low'])
    rule347 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule348 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule349 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['low'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes']) 
    rule350 = ctrl.Rule(C6['high'] & C9['low'] & C1['high'] & C2['high'] & C3['high'] & C4['high'] & 
               C5['low'] & C7['low'] & C8['high'], edge['yes'])
    rule351 = ctrl.Rule(C6['high'] & C9['high'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['yes'])
    rule352 = ctrl.Rule(C6['low'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule353 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['high'], edge['high'])
    rule354 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['low'], edge['high'])
    rule355 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['low'] & C7['high'] & C8['high'], edge['high'])
    rule356 = ctrl.Rule(C6['low'] & C9['high'] & C1['high'] & C2['low'] & C3['high'] & C4['low'] & 
               C5['high'] & C7['high'] & C8['high'], edge['yes'])
    rule357 = ctrl.Rule(C6['high'] & C9['low'] & C1['low'] & C2['high'] & C3['low'] & C4['high'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    rule358 = ctrl.Rule(C6['low'] & C9['low'] & C1['high'] & C2['high'] & C3['low'] & C4['low'] & 
               C5['high'] & C7['low'] & C8['low'], edge['yes'])
    

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7,rule8, rule9, rule10, rule11, rule12,
        rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, 
        rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, 
        rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48, 
        rule49, rule50, rule51, rule52, rule53, rule54, rule55, rule56, rule57, rule58, rule59, rule60, 
        rule61, rule62, rule63, rule64, rule65, rule66, rule67, rule68, rule69, rule70, rule71, rule72, 
        rule73, rule74, rule75, rule76, rule77, rule78, rule79, rule80, rule81, rule82, rule83, rule84, 
        rule85, rule86, rule87, rule88, rule89, rule90, rule91, rule92, rule93, rule94, rule95, rule96, 
        rule97, rule98, rule99, rule100, rule101, rule102, rule103, rule104, rule105, rule106, rule107, 
        rule108, rule109, rule110, rule111, rule112, rule113, rule114, rule115, rule116, rule117, rule118,
        rule119, rule120, rule121, rule122, rule123, rule124, rule125, rule126, rule127, rule128, rule129,
        rule130, rule131, rule132, rule133, rule134, rule135, rule136, rule137, rule138, rule139, rule140,
        rule141, rule142, rule143, rule144, rule145, rule146, rule147, rule148, rule149, rule150, rule151,
        rule152, rule153, rule154, rule155, rule156, rule157, rule158, rule159, rule160, rule161, rule162,
        rule163, rule164, rule165, rule166, rule167, rule168, rule169, rule170, rule171, rule172, rule173,
        rule174, rule175, rule176, rule177, rule178, rule179, rule180, rule181, rule182, rule183, rule184,
        rule185, rule186, rule187, rule188, rule189, rule190, rule191, rule192, rule193, rule194, rule195,
        rule196, rule197, rule198, rule199, rule200, rule201, rule202, rule203, rule204, rule205, rule206, 
        rule207, rule208, rule209, rule210, rule211, rule212, rule213, rule214, rule215, rule216, rule217,
        rule218, rule219, rule220, rule221, rule222, rule223, rule224, rule225, rule226, rule227, rule228,
        rule229, rule230, rule231, rule232, rule233, rule234, rule235, rule236, rule237, rule238, rule239,
        rule240, rule241, rule242, rule243, rule244, rule245, rule246, rule247, rule248, rule249, rule250,
        rule251, rule252, rule253, rule254, rule255, rule256, rule257, rule258, rule259, rule260, rule261, 
        rule262, rule263, rule264, rule265, rule266, rule267, rule268, rule269, rule270, rule271, rule272,
        rule273, rule274, rule275, rule276, rule277, rule278, rule279, rule280, rule281, rule282, rule283,
        rule284, rule285, rule286, rule287, rule288, rule289, rule290, rule291, rule292, rule293, rule294,
        rule295, rule296, rule297, rule298, rule299, rule300, rule301, rule302, rule303, rule304, rule305,
        rule306, rule307, rule308, rule309, rule310, rule311, rule312, rule313, rule314, rule315, rule316,
        rule317, rule318, rule319, rule320, rule321, rule322, rule323, rule324, rule325, rule326, rule327,
        rule328, rule329, rule330, rule331, rule332, rule333, rule334, rule335, rule336, rule337, rule338,
        rule339, rule340, rule341, rule342, rule343, rule344, rule345, rule346, rule347, rule348, rule349,
        rule350, rule351, rule352, rule353, rule354, rule355, rule356, rule357, rule358]
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
        print(f"Procesando: {(i*cols)/(cols*rows)*100}%     ",end="\r")
        for j in range(1, cols-1):
            neighbor_values = [fuzzy_image[i+di, j+dj] for di, dj in neighbors]   
            edge_detect.input['C1'] = neighbor_values[0]
            # print(i)
            # print(j)
            # print(neighbor_values[0])
            edge_detect.input['C2'] = neighbor_values[1]
            # print(neighbor_values[1])
            edge_detect.input['C3'] = neighbor_values[2]
            # print(neighbor_values[2])
            edge_detect.input['C4'] = neighbor_values[3]
            # print(neighbor_values[3])
            edge_detect.input['C5'] = neighbor_values[4 ]
            # print(neighbor_values[4])
            edge_detect.input['C6'] = neighbor_values[5]
            # print(neighbor_values[5])
            edge_detect.input['C7'] = neighbor_values[6]
            # print(neighbor_values[6])
            edge_detect.input['C8'] = neighbor_values[7]
            # print(neighbor_values[7])
            edge_detect.input['C9'] = neighbor_values[8]
            # print(neighbor_values[8])

            try:
                edge_detect.compute()            
            except Exception as e:
                print(f"Error en la posición ({i}, {j}): {e}\n\n")
                #edge_image[i, j] =  fuzzy_image[i, j]
                edge_image[i, j] =  0
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
    image2=filter_h(image)
    imagen=cv2.resize(image2, (200, int(x*200/y)))
    #gray=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Original Image', image2)
    cv2.waitKey(0)

    # Aplicar detección de bordes difusa
    fuzzy_image = image2.astype(float) / 256.00000000  # Normalizar la imagen entre 0 y 1
    edge_detect = create_fuzzy_system (image2)
    edge_image = apply_fuzzy_rules_to_image (fuzzy_image, edge_detect)
    #print(edge_image)
    edge_image_uint8 = (edge_image * 255).astype(np.uint8)
    print(edge_image_uint8)
    # Definir el valor umbral
    umbral = 127  # Puedes ajustar este valor según sea necesario
    tono_deseado = 0.5

    # Crear una máscara para seleccionar los píxeles con el valor deseado
    umbral_inferior = tono_deseado   # Ajusta este valor según sea necesario
    umbral_superior = tono_deseado   # Ajusta este valor según sea necesario
    mascara = cv2.inRange(edge_image_uint8, umbral_inferior, umbral_superior)

    # Crear una imagen para resaltar los píxeles seleccionados
    imagen_resaltada = np.zeros_like(edge_image_uint8)
    imagen_resaltada[mascara > 0] = 1  # Resaltar con blanco

    # Convertir la imagen resaltada al rango [0, 255]
    imagen_resaltada = (imagen_resaltada * 255).astype(np.uint8)


    # Aplicar el umbral negro
    _, imagen_umbral = cv2.threshold(edge_image_uint8, umbral, 255, cv2.THRESH_BINARY)
    # Mostrar resultados
    # cv2.imshow('Original Image', image)
    cv2.imshow('Fuzzy Edge Detected Image', imagen_umbral)  # Escalar a 0-255 para visualizar
    cv2.imshow('Solo contorno', edge_image_uint8)
    cv2.waitKey(0)
    #cv2.imwrite("test.jpeg",edge_image_uint8)
    cv2.destroyAllWindows()

# Ruta a la imagen

main()