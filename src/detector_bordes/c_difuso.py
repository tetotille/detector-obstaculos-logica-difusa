import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from os.path import dirname, abspath, join
from sys import argv
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from src.detector_hsv.hsv_filter import filter_h

# Definir las funciones de membresía para los píxeles vecinos y el píxel central
def define_membership_functions(image):
    x,y = image.shape
    #imagen=cv2.resize(image, (200, int(x*200/y)))
    #gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_pixel = np.min(image)/256
    max_pixel = np.max(image)/256
    print(min_pixel, max_pixel)
    universe = np.arange(min_pixel, max_pixel, (1/256))  # Normalización entre 0 y 1
    valor_medio = ((min_pixel + max_pixel)/2)
    print(valor_medio)
    print(universe)
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
        C.automf(3, names=['low','medium', 'high']) # Dividir en 2 funciones de membresía

    # Definir el consecuente (edge) con un rango adecuado
    edge = ctrl.Consequent(universe2, 'edge')

    # Definir funciones de membresía para cada vecino
    for C in [C1, C2, C3, C4, C5, C6, C7, C8, C9]:
        C['low'] = fuzz.trimf(universe, [min_pixel, min_pixel, 0.5])
        C['medium'] = fuzz.trimf(universe, [(max_pixel/3), valor_medio, max_pixel*(2/3)])
        C['high'] = fuzz.trimf(universe, [0.5, max_pixel, max_pixel])
    
    edge['low'] = fuzz.trimf(universe2, [0, 0, 0.5])
    edge['high']= fuzz.trimf(universe2, [0.5, 1, 1])
    edge['yes'] = fuzz.trimf(universe2, [0.5, 0.5, 0.5])
    
        # Definir el universo de discurso
    universe = np.arange(0, 1.01, 0.01)
    universe2 = np.arange(0, 1.01, 0.01)

    ##### Se hace una copia exclusivamente para dibujar y ver las funciones de membresía #####
    # Definir las funciones de membresía
    C_low = fuzz.trimf(universe, [0, 0, (1/3)])
    C_medium = fuzz.trimf(universe, [(1/3), 0.5, (2/3)])
    C_high = fuzz.trimf(universe, [(2/3), 1, 1])

    edge_low = fuzz.trimf(universe2, [0, 0, 0.5])
    edge_high= fuzz.trimf(universe2, [0.5, 1, 1])
    edge_yes = fuzz.trimf(universe2, [0.5, 0.5, 0.5])

    # Graficar las funciones de membresía
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Funciones de membresía para C (low y high)
    ax[0].plot(universe, C_low, 'b', linewidth=1.5, label='Low')
    ax[0].plot(universe, C_medium, 'c', linewidth=1.5, label='Low')
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

def calculate_membership_values(fuzzy_image, membership_functions):
    membership_values = []
    for func in membership_functions:
        membership_values.append(fuzz.interp_membership(fuzzy_image, func.universe, func.mf))
    return np.array(membership_values)

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
    
    rule10 = ctrl.Rule(C2['low'] & C5['low'] & C9['low'] & 
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
                      C4['high'] & C6['high'] & C8['high'], edge['low'])
    
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
                      C1['low'] & C2['low'] & C3['low'], edge['low'])  
    
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
    rule23 = ctrl.Rule(C4['high'] & C5['high'] & C1['high'] &   #cambiar
                     C6['low'] & C8['low'] & C9['low'] & 
                     C2['high'] & C3['low'] & C7['low'], edge['yes'])
    
    rule24 = ctrl.Rule(C5['high'] & C6['high'] & C1['high'] &  
                     C4['low'] & C8['low'] & C7['low'] & 
                     C2['high'] & C3['high'] & C9['high'], edge['yes'])
    
    rule25 = ctrl.Rule(C3['high'] & C5['high'] & C9['high'] & 
                     C1['low'] & C2['low'] & C4['low'] & 
                     C6['high'] & C8['high'] & C7['high'], edge['yes'])
    
    rule26 = ctrl.Rule(C1['high'] & C5['high'] & C4['high'] & 
                     C2['high'] & C6['low'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])
    
    rule27 = ctrl.Rule(C1['low'] & C5['high'] & C4['high'] & 
                     C2['low'] & C6['high'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['low'])
    
    rule28 = ctrl.Rule(C1['low'] & C5['low'] & C4['high'] &     #cambiar
                     C2['low'] & C6['high'] & C3['low'] & 
                     C8['high'] & C9['high'] & C7['high'], edge['yes'])
    
    rule29= ctrl.Rule(C1['low'] & C5['high'] & C4['low'] & 
                     C2['high'] & C6['low'] & C3['low'] & 
                     C8['high'] & C9['low'] & C7['low'], edge['low'])   
    
    rule30= ctrl.Rule(C1['high'] & C5['low'] & C4['high'] & 
                     C2['low'] & C6['high'] & C3['high'] & 
                     C8['low'] & C9['high'] & C7['high'], edge['low'])   
    
    rule31 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] & 
                      C4['high'] & C5['low'] & C6['high'] & 
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])
    

    rule32 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] & 
                      C4['high'] & C5['low'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['low'], edge['yes'])
    # Definir la regla para cuando solo una variable es 'low' y el resto son 'high'
    rule33 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] & 
                      C4['low'] & C5['low'] & C6['high'] & 
                      C7['low'] & C8['high'] & C9['high'], edge['yes'])

# Definir la regla para cuando solo una variable es 'high' y el resto son 'low'
    rule34 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] & 
                      C4['high'] & C5['high'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule35 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])

    rule36 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule37 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])

    rule38 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule39 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    
    rule40 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule41 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule42 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    
    rule43 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    
    rule44 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    
    rule45 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    
    rule46 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    
    rule47 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule48 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule49 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    
    rule50 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    
    rule51 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule52 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    

    rules1 = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12,
        rule15, rule16, rule17,  rule19, rule20, rule21, rule22, rule23, rule24, 
        rule25, rule26,  rule28,  rule31, rule32, rule33, rule34, rule35, rule36, 
        rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46,
        rule47, rule48, rule49, rule50, rule51, rule52]
    
    
    rule53 = ctrl.Rule(C1['high'] & C3['high'] & C5['high'] & 
                      C2['high'] & C4['high'] & C6['high'] &
                      C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule54 = ctrl.Rule(C5['high'] & C7['high'] & C9['high'] &
                      C1['medium'] & C2['medium'] & C3['medium'] &
                      C4['high'] & C6['high'] & C8['high'], edge['yes'])

    rule55 = ctrl.Rule(C5['high'] & C4['medium'] & C1['medium'] &
                      C9['high'] & C2['high'] & C7['medium'] &
                      C3['high'] & C6['high'] & C8['high'], edge['yes'])

    rule56 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] &
                      C9['medium'] & C6['medium'] & C3['medium'] &
                      C5['high'] & C7['high'] & C8['high'], edge['yes'])

    rule57 = ctrl.Rule(C1['medium'] & C2['medium'] & C4['medium'] &
                      C5['medium'] & C6['high'] & C3['high'] &
                      C7['high'] & C9['high'] & C8['high'], edge['yes'])

    rule58 = ctrl.Rule(C2['high'] & C1['high'] & C3['high'] &
                      C8['medium'] & C6['high'] & C9['high'] &
                      C4['medium'] & C5['medium'] & C7['medium'], edge['yes'])

    rule59 = ctrl.Rule(C4['high'] & C7['high'] & C8['high'] &
                     C1['high'] & C6['medium'] & C3['medium'] &
                     C2['medium'] & C9['high'] & C5['medium'], edge['yes'])

    rule60 = ctrl.Rule(C6['medium'] & C5['medium'] & C8['medium'] &
                     C1['high'] & C4['high'] & C3['high'] &
                     C2['high'] & C9['medium'] & C7['high'], edge['yes'])

    rule61 = ctrl.Rule(C4['medium'] & C5['medium'] & C1['medium'] &
                     C6['high'] & C8['high'] & C9['high'] &
                     C2['medium'] & C3['medium'] & C7['medium'], edge['yes'])

    rule62 = ctrl.Rule(C2['medium'] & C5['medium'] & C9['medium'] &
                     C1['medium'] & C6['medium'] & C3['medium'] &
                     C4['high'] & C8['high'] & C7['high'], edge['yes'])

    rule63 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] &
                     C5['medium'] & C6['medium'] & C3['medium'] &
                     C8['medium'] & C9['medium'] & C7['medium'], edge['yes'])

    # ... añadir las otras reglas aquí

    rule64 = ctrl.Rule(C1['medium'] & C5['medium'] & C9['medium'] &
                       C2['high'] & C3['high'] & C4['medium'] &
                       C6['high'] & C7['medium'] & C8['medium'], edge['yes'])

    rule65 =ctrl.Rule(C1['medium'] & C5['medium'] & C9['medium'] &
                       C2['medium'] & C3['medium'] & C4['medium'] &
                       C6['medium'] & C7['medium'] & C8['medium'], edge['low'])

    rule66 = ctrl.Rule(C5['high'] & C3['high'] & C1['high'] &
                      C9['high'] & C2['high'] & C7['high'] &
                      C4['high'] & C6['high'] & C8['high'], edge['low'])

    rule67 = ctrl.Rule(C5['medium'] & C3['medium'] & C1['medium'] &
                      C2['medium'] & C4['medium'] & C6['medium'] &
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule68= ctrl.Rule(C5['medium'] & C3['medium'] & C6['medium'] &
                      C9['medium'] & C8['medium'] & C2['medium'] &
                      C1['high'] & C4['high'] & C7['high'], edge['yes'])

    rule69= ctrl.Rule(C5['medium'] & C4['medium'] & C6['medium'] &
                      C9['medium'] & C8['medium'] & C7['medium'] &
                      C1['high'] & C2['high'] & C3['high'], edge['yes'])

    rule70 = ctrl.Rule(C9['high'] & C3['high'] & C6['high'] &
                      C7['medium'] & C8['medium'] & C5['medium'] &
                      C1['medium'] & C2['medium'] & C3['medium'], edge['low'])

    #otro tipo de division

    rule71 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] &
                      C5['high'] & C6['medium'] & C3['medium'] &
                      C7['medium'] & C9['medium'] & C8['medium'], edge['yes'])

    rule72 = ctrl.Rule(C2['medium'] & C1['medium'] & C3['medium'] &
                      C8['high'] & C6['medium'] & C9['medium'] &
                      C4['high'] & C5['high'] & C7['high'], edge['yes'])

    rule73 = ctrl.Rule(C4['medium'] & C7['medium'] & C8['medium'] &
                     C1['medium'] & C6['high'] & C3['high'] &
                     C2['high'] & C9['medium'] & C5['high'], edge['yes'])

    rule73 = ctrl.Rule(C6['high'] & C5['high'] & C8['high'] &
                     C1['medium'] & C4['medium'] & C3['medium'] &
                     C2['medium'] & C9['high'] & C7['medium'], edge['yes'])
    #Aca otro
    rule74 = ctrl.Rule(C4['high'] & C5['high'] & C1['high'] &   #cambiar
                     C6['medium'] & C8['medium'] & C9['medium'] &
                     C2['high'] & C3['medium'] & C7['medium'], edge['yes'])

    rule75 = ctrl.Rule(C5['high'] & C6['high'] & C1['high'] &
                     C4['medium'] & C8['medium'] & C7['medium'] &
                     C2['high'] & C3['high'] & C9['high'], edge['yes'])

    rule76 = ctrl.Rule(C3['high'] & C5['high'] & C9['high'] &
                     C1['medium'] & C2['medium'] & C4['medium'] &
                     C6['high'] & C8['high'] & C7['high'], edge['yes'])

    rule77 = ctrl.Rule(C1['high'] & C5['high'] & C4['high'] &
                     C2['high'] & C6['medium'] & C3['medium'] &
                     C8['medium'] & C9['medium'] & C7['medium'], edge['yes'])

    rule78 = ctrl.Rule(C1['medium'] & C5['high'] & C4['high'] &
                     C2['medium'] & C6['high'] & C3['medium'] &
                     C8['medium'] & C9['medium'] & C7['medium'], edge['low'])

    rule79 = ctrl.Rule(C1['medium'] & C5['medium'] & C4['high'] &     #cambiar
                     C2['medium'] & C6['high'] & C3['medium'] &
                     C8['high'] & C9['high'] & C7['high'], edge['yes'])

    rule80= ctrl.Rule(C1['medium'] & C5['high'] & C4['medium'] &
                     C2['high'] & C6['medium'] & C3['medium'] &
                     C8['high'] & C9['medium'] & C7['medium'], edge['low'])

    rule81= ctrl.Rule(C1['high'] & C5['medium'] & C4['high'] &
                     C2['medium'] & C6['high'] & C3['high'] &
                     C8['medium'] & C9['high'] & C7['high'], edge['low'])

    rule82 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                      C4['high'] & C5['medium'] & C6['high'] &
                      C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])


    rule83 = ctrl.Rule(C1['high'] & C2['high'] & C3['medium'] &
                      C4['high'] & C5['medium'] & C6['medium'] &
                      C7['high'] & C8['high'] & C9['medium'], edge['yes'])
    # Definir la regla para cuando solo una variable es 'medium' y el resto son 'high'
    rule84 = ctrl.Rule(C1['medium'] & C2['high'] & C3['high'] &
                      C4['medium'] & C5['medium'] & C6['high'] &
                      C7['medium'] & C8['high'] & C9['high'], edge['yes'])

# Definir la regla para cuando solo una variable es 'high' y el resto son 'medium'
    rule85= ctrl.Rule(C1['high'] & C2['medium'] & C3['medium'] &
                      C4['high'] & C5['high'] & C6['medium'] &
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule86 = ctrl.Rule(C1['high'] & C2['medium'] & C3['medium'] &
                   C4['high'] & C5['high'] & C6['medium'] &
                   C7['high'] & C8['medium'] & C9['medium'], edge['yes'])

    rule87 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['medium'] & C5['high'] & C6['medium'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule88 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['high'] &
                   C4['medium'] & C5['high'] & C6['high'] &
                   C7['medium'] & C8['medium'] & C9['high'], edge['yes'])

    rule89 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['medium'] & C5['high'] & C6['medium'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule90 = ctrl.Rule(C1['high'] & C2['medium'] & C3['medium'] &
                   C4['high'] & C5['medium'] & C6['medium'] &
                   C7['high'] & C8['high'] & C9['medium'], edge['yes'])

    rule91 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['high'] & C5['medium'] & C6['medium'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule92 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['medium'] & C5['medium'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule93 = ctrl.Rule(C1['medium'] & C2['high'] & C3['high'] &
                   C4['medium'] & C5['medium'] & C6['high'] &
                   C7['medium'] & C8['medium'] & C9['high'], edge['yes'])

    rule94 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['medium'] & C5['medium'] & C6['high'] &
                   C7['medium'] & C8['medium'] & C9['high'], edge['yes'])

    rule95 = ctrl.Rule(C1['high'] & C2['high'] & C3['medium'] &
                   C4['high'] & C5['medium'] & C6['medium'] &
                   C7['high'] & C8['medium'] & C9['medium'], edge['yes'])

    rule96 = ctrl.Rule(C1['medium'] & C2['high'] & C3['high'] &
                   C4['medium'] & C5['high'] & C6['high'] &
                   C7['medium'] & C8['medium'] & C9['high'], edge['yes'])

    rule97 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['high'] &
                   C4['medium'] & C5['high'] & C6['high'] &
                   C7['medium'] & C8['high'] & C9['high'], edge['yes'])

    rule98 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['medium'] & C5['high'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule99 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['high'] & C5['high'] & C6['medium'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule100 = ctrl.Rule(C1['high'] & C2['medium'] & C3['medium'] &
                   C4['high'] & C5['high'] & C6['medium'] &
                   C7['high'] & C8['high'] & C9['medium'], edge['yes'])

    rule101 = ctrl.Rule(C1['high'] & C2['high'] & C3['medium'] &
                   C4['high'] & C5['high'] & C6['medium'] &
                   C7['high'] & C8['medium'] & C9['medium'], edge['yes'])

    rule102 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['medium'] & C5['high'] & C6['high'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule103 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['high'] & C6['medium'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])
    
    rules2 = [rule53, rule54, rule55, rule56, rule57, rule58, rule59, rule60, rule61, rule62, rule63, rule64,
        rule67, rule68, rule69,  rule71, rule72, rule73, rule74, rule75, 
        rule76, rule77, rule79, rule82, rule83, rule84, rule85, rule86, 
        rule87, rule88, rule89, rule90, rule91, rule92, rule93, rule94, rule95, rule96,
        rule97, rule98, rule99, rule100, rule101, rule102, rule103]
    
    rule104 = ctrl.Rule(C1['medium'] & C3['medium'] & C5['medium'] & 
                      C2['medium'] & C4['medium'] & C6['medium'] &
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])

    rule105 = ctrl.Rule(C5['medium'] & C7['medium'] & C9['medium'] &
                      C1['low'] & C2['low'] & C3['low'] &
                      C4['medium'] & C6['medium'] & C8['medium'], edge['yes'])

    rule106 = ctrl.Rule(C5['medium'] & C4['low'] & C1['low'] &
                      C9['medium'] & C2['medium'] & C7['low'] &
                      C3['medium'] & C6['medium'] & C8['medium'], edge['yes'])

    rule107 = ctrl.Rule(C1['medium'] & C2['medium'] & C4['medium'] &
                      C9['low'] & C6['low'] & C3['low'] &
                      C5['medium'] & C7['medium'] & C8['medium'], edge['yes'])

    rule108 = ctrl.Rule(C1['low'] & C2['low'] & C4['low'] &
                      C5['low'] & C6['medium'] & C3['medium'] &
                      C7['medium'] & C9['medium'] & C8['medium'], edge['yes'])

    rule109 = ctrl.Rule(C2['medium'] & C1['medium'] & C3['medium'] &
                      C8['low'] & C6['medium'] & C9['medium'] &
                      C4['low'] & C5['low'] & C7['low'], edge['yes'])

    rule110 = ctrl.Rule(C4['medium'] & C7['medium'] & C8['medium'] &
                     C1['medium'] & C6['low'] & C3['low'] &
                     C2['low'] & C9['medium'] & C5['low'], edge['yes'])

    rule111 = ctrl.Rule(C6['low'] & C5['low'] & C8['low'] &
                     C1['medium'] & C4['medium'] & C3['medium'] &
                     C2['medium'] & C9['low'] & C7['medium'], edge['yes'])

    rule112 = ctrl.Rule(C4['low'] & C5['low'] & C1['low'] &
                     C6['medium'] & C8['medium'] & C9['medium'] &
                     C2['low'] & C3['low'] & C7['low'], edge['yes'])

    rule113 = ctrl.Rule(C2['low'] & C5['low'] & C9['low'] &
                     C1['low'] & C6['low'] & C3['low'] &
                     C4['medium'] & C8['medium'] & C7['medium'], edge['yes'])

    rule114 = ctrl.Rule(C1['medium'] & C2['medium'] & C4['medium'] &
                     C5['low'] & C6['low'] & C3['low'] &
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])

    # ... añadir las otras reglas aquí

    rule115 = ctrl.Rule(C1['low'] & C5['low'] & C9['low'] &
                       C2['medium'] & C3['medium'] & C4['low'] &
                       C6['medium'] & C7['low'] & C8['low'], edge['yes'])

    rule116 =ctrl.Rule(C1['low'] & C5['low'] & C9['low'] &
                       C2['low'] & C3['low'] & C4['low'] &
                       C6['low'] & C7['low'] & C8['low'], edge['low'])

    rule117 = ctrl.Rule(C5['medium'] & C3['medium'] & C1['medium'] &
                      C9['medium'] & C2['medium'] & C7['medium'] &
                      C4['medium'] & C6['medium'] & C8['medium'], edge['low'])

    rule118 = ctrl.Rule(C5['low'] & C3['low'] & C1['low'] &
                      C2['low'] & C4['low'] & C6['low'] &
                      C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule119= ctrl.Rule(C5['low'] & C3['low'] & C6['low'] &
                      C9['low'] & C8['low'] & C2['low'] &
                      C1['medium'] & C4['medium'] & C7['medium'], edge['yes'])

    rule120= ctrl.Rule(C5['low'] & C4['low'] & C6['low'] &
                      C9['low'] & C8['low'] & C7['low'] &
                      C1['medium'] & C2['medium'] & C3['medium'], edge['yes'])

    rule121 = ctrl.Rule(C9['medium'] & C3['medium'] & C6['medium'] &
                      C7['low'] & C8['low'] & C5['low'] &
                      C1['low'] & C2['low'] & C3['low'], edge['low'])

    #otro tipo de division

    rule122 = ctrl.Rule(C1['medium'] & C2['medium'] & C4['medium'] &
                      C5['medium'] & C6['low'] & C3['low'] &
                      C7['low'] & C9['low'] & C8['low'], edge['yes'])

    rule123 = ctrl.Rule(C2['low'] & C1['low'] & C3['low'] &
                      C8['medium'] & C6['low'] & C9['low'] &
                      C4['medium'] & C5['medium'] & C7['medium'], edge['yes'])

    rule124 = ctrl.Rule(C4['low'] & C7['low'] & C8['low'] &
                     C1['low'] & C6['medium'] & C3['medium'] &
                     C2['medium'] & C9['low'] & C5['medium'], edge['yes'])

    rule125 = ctrl.Rule(C6['medium'] & C5['medium'] & C8['medium'] &
                     C1['low'] & C4['low'] & C3['low'] &
                     C2['low'] & C9['medium'] & C7['low'], edge['yes'])
    #Aca otro
    rule126 = ctrl.Rule(C4['medium'] & C5['medium'] & C1['medium'] &   #cambiar
                     C6['low'] & C8['low'] & C9['low'] &
                     C2['medium'] & C3['low'] & C7['low'], edge['yes'])

    rule127 = ctrl.Rule(C5['medium'] & C6['medium'] & C1['medium'] &
                     C4['low'] & C8['low'] & C7['low'] &
                     C2['medium'] & C3['medium'] & C9['medium'], edge['yes'])

    rule128 = ctrl.Rule(C3['medium'] & C5['medium'] & C9['medium'] &
                     C1['low'] & C2['low'] & C4['low'] &
                     C6['medium'] & C8['medium'] & C7['medium'], edge['yes'])

    rule129 = ctrl.Rule(C1['medium'] & C5['medium'] & C4['medium'] &
                     C2['medium'] & C6['low'] & C3['low'] &
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])

    rule130 = ctrl.Rule(C1['low'] & C5['medium'] & C4['medium'] &
                     C2['low'] & C6['medium'] & C3['low'] &
                     C8['low'] & C9['low'] & C7['low'], edge['low'])

    rule131 = ctrl.Rule(C1['low'] & C5['low'] & C4['medium'] &     #cambiar
                     C2['low'] & C6['medium'] & C3['low'] &
                     C8['medium'] & C9['medium'] & C7['medium'], edge['yes'])

    rule132= ctrl.Rule(C1['low'] & C5['medium'] & C4['low'] &
                     C2['medium'] & C6['low'] & C3['low'] &
                     C8['medium'] & C9['low'] & C7['low'], edge['low'])

    rule133= ctrl.Rule(C1['medium'] & C5['low'] & C4['medium'] &
                     C2['low'] & C6['medium'] & C3['medium'] &
                     C8['low'] & C9['medium'] & C7['medium'], edge['low'])

    rule134 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                      C4['medium'] & C5['low'] & C6['medium'] &
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])


    rule135 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['low'] &
                      C4['medium'] & C5['low'] & C6['low'] &
                      C7['medium'] & C8['medium'] & C9['low'], edge['yes'])
    # Definir la regla para cuando solo una variable es 'low' y el resto son 'medium'

    rule136 = ctrl.Rule(C1['low'] & C2['medium'] & C3['medium'] &
                      C4['low'] & C5['low'] & C6['medium'] &
                      C7['low'] & C8['medium'] & C9['medium'], edge['yes'])

# Definir la regla para cuando solo una variable es 'medium' y el resto son 'low'
    rule137 = ctrl.Rule(C1['medium'] & C2['low'] & C3['low'] &
                      C4['medium'] & C5['medium'] & C6['low'] &
                      C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule138 = ctrl.Rule(C1['medium'] & C2['low'] & C3['low'] &
                   C4['medium'] & C5['medium'] & C6['low'] &
                   C7['medium'] & C8['low'] & C9['low'], edge['yes'])

    rule139 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['medium'] & C6['low'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule140 = ctrl.Rule(C1['low'] & C2['low'] & C3['medium'] &
                   C4['low'] & C5['medium'] & C6['medium'] &
                   C7['low'] & C8['low'] & C9['medium'], edge['yes'])

    rule141 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['low'] & C5['medium'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])

    rule142 = ctrl.Rule(C1['medium'] & C2['low'] & C3['low'] &
                   C4['medium'] & C5['low'] & C6['low'] &
                   C7['medium'] & C8['medium'] & C9['low'], edge['yes'])

    rule143 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['medium'] & C5['low'] & C6['low'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule144 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['low'] & C6['medium'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule145 = ctrl.Rule(C1['low'] & C2['medium'] & C3['medium'] &
                   C4['low'] & C5['low'] & C6['medium'] &
                   C7['low'] & C8['low'] & C9['medium'], edge['yes'])

    rule146 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['low'] & C5['low'] & C6['medium'] &
                   C7['low'] & C8['low'] & C9['medium'], edge['yes'])

    rule147 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['low'] &
                   C4['medium'] & C5['low'] & C6['low'] &
                   C7['medium'] & C8['low'] & C9['low'], edge['yes'])

    rule148 = ctrl.Rule(C1['low'] & C2['medium'] & C3['medium'] &
                   C4['low'] & C5['medium'] & C6['medium'] &
                   C7['low'] & C8['low'] & C9['medium'], edge['yes'])

    rule149 = ctrl.Rule(C1['low'] & C2['low'] & C3['medium'] &
                   C4['low'] & C5['medium'] & C6['medium'] &
                   C7['low'] & C8['medium'] & C9['medium'], edge['yes'])

    rule150 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['medium'] & C6['medium'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule151 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['medium'] & C5['medium'] & C6['low'] &
                   C7['medium'] & C8['medium'] & C9['medium'], edge['yes'])

    rule152 = ctrl.Rule(C1['medium'] & C2['low'] & C3['low'] &
                   C4['medium'] & C5['medium'] & C6['low'] &
                   C7['medium'] & C8['medium'] & C9['low'], edge['yes'])

    rule153 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['low'] &
                   C4['medium'] & C5['medium'] & C6['low'] &
                   C7['medium'] & C8['low'] & C9['low'], edge['yes'])

    rule154 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['low'] & C5['medium'] & C6['medium'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])

    rule155 = ctrl.Rule(C1['medium'] & C2['medium'] & C3['medium'] &
                   C4['medium'] & C5['medium'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rules3 = [rule104, rule105, rule106, rule107, rule108, rule109, rule110, rule111, rule112, rule113, rule114, rule115,
        rule118, rule119, rule120,  rule122, rule123, rule124, rule125, rule126, rule127, 
        rule128, rule129,  rule131,  rule134, rule135, rule136, rule137, rule138, rule139, 
        rule140, rule141, rule142, rule143, rule144, rule145, rule146, rule147, rule148, rule149,
        rule150, rule151, rule152, rule153, rule154, rule155]
    
    rules=rules1+rules2+rules3
    
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
        filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
    image=cv2.imread(filename)
    x,y,a = image.shape
    #image2=filter_h(image)
    imagen=cv2.resize(image, (200, int(x*200/y)))
    gray=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Original Image', gray)
    cv2.waitKey(0)

    # Aplicar detección de bordes difusa
    fuzzy_image = gray.astype(float) / 256.00000000  # Normalizar la imagen entre 0 y 1
    edge_detect = create_fuzzy_system (gray)
    edge_image = apply_fuzzy_rules_to_image (fuzzy_image, edge_detect)
    
    #print(edge_image)
    edge_image_uint8 = (edge_image * 255).astype(np.uint8)
    print(edge_image_uint8)
    # Definir el valor umbral
    umbral = 127  # Puedes ajustar este valor según sea necesario
    tono_deseado = 0.5

    # Crear una máscara para seleccionar los píxeles con el valor deseado
    umbral_inferior = tono_deseado-(1/255)   # Ajusta este valor según sea necesario
    umbral_superior = tono_deseado+(1/255)   # Ajusta este valor según sea necesario
    mascara = cv2.inRange(edge_image_uint8, umbral_inferior, umbral_superior)

    # Crear una imagen para resaltar los píxeles seleccionados
    imagen_resaltada = np.zeros_like(edge_image_uint8)
    imagen_resaltada[mascara > 0] = 1  # Resaltar con blanco

    # Convertir la imagen resaltada al rango [0, 255]
    imagen_resaltada = (imagen_resaltada * 255).astype(np.uint8)

    # Aplicar el umbral negro
    _, imagen_umbral = cv2.threshold(edge_image_uint8, 1, 255, cv2.THRESH_BINARY)

    # Suavizar la máscara de contornos
    mask_smooth = cv2.GaussianBlur(imagen_umbral, (5, 5), 0)

# Detectar contornos
    #contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mostrar resultados
    # cv2.imshow('Original Image', image)
    #cv2.imshow('Fuzzy Edge Detected Image', mask_smooth)  # Escalar a 0-255 para visualizar
    cv2.imshow('Solo contorno', edge_image_uint8)
    cv2.waitKey(0)
    cv2.imwrite("tes.jpeg",imagen_umbral)
    cv2.imwrite("filename.png", imagen_umbral)
    cv2.destroyAllWindows()

# Ruta a la imagen

main() 