
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from os.path import dirname, abspath, join
from sys import argv
import cv2

def threshold_image(I, Th):
    """
    Umbraliza la imagen I con el umbral Th.
    Devuelve una nueva imagen Bh.
    """
    Bh = np.zeros_like(I)
    Bh[I >= Th] = I[I >= Th]
    return Bh

def calculate_standard_deviation(Bh):
    """
    Calcula la desviación estándar de las intensidades de píxeles de la imagen Bh.
    """
    return np.std(Bh)

def mapping(I, Cpa_minus_1, Cpa):
    """
    Mapea la intensidad de la imagen del dominio espacial [gmin, gmax] al dominio difuso [0, 1].
    """
    fuzzy_image = np.zeros_like(I, dtype=float)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I[i, j] >= Cpa_minus_1 and I[i, j] <= Cpa:
                fuzzy_image[i, j] = (I[i, j] - Cpa_minus_1) / (Cpa - Cpa_minus_1)
            elif I[i, j] > Cpa:
                fuzzy_image[i, j] = 1
            else:
                fuzzy_image[i, j] = 0
    return fuzzy_image

def calculate_RBEM(Cp0, Cp1, Cp2, Cp3, psi_L, psi_CO, psi_TH, psi_S, D_L, D_CO, D_TH, D_S):
    """
    Calcula RBEM basado en los puntos de corte Cp0, Cp1, Cp2 y Cp3.
    """
    RBEM = (psi_L * (1 - D_L) +
            psi_CO * (1 - D_CO) +
            psi_TH * (1 - D_TH) +
            psi_S * (1 - D_S))
    return RBEM

def classify_fuzzy_values(fuzzy_image):
    """
    Clasifica los valores de membresía difusa en alta (>= 0.5) y baja (< 0.5).
    """
    high_membership = fuzzy_image >= 0.5
    low_membership = fuzzy_image < 0.5
    return high_membership, low_membership

def main(I, gmin, gmax):
    SD_values = []
    thresholded_images = []
    
    # Paso 1: Umbralizar la imagen con diferentes valores de Th
    for Th in range(gmin, gmax + 1):
        Bh = threshold_image(I, Th)
        thresholded_images.append(Bh)
        
        # Paso 2: Calcular y guardar la desviación estándar de cada imagen umbralizada
        SD = calculate_standard_deviation(Bh)
        SD_values.append(SD)
    
    # Paso 3: Obtener SD1 y SDmax
    SD1 = SD_values[0]
    SDmax = max(SD_values)
    
    best_RBEMk = float('inf')
    best_mk = None
    membership_matrix = []

    # Parámetros arbitrarios (ajustar según sea necesario)
    psi_L = 1.0
    psi_CO = 1.0
    psi_TH = 1.0
    psi_S = 1.0
    D_L = 0.5
    D_CO = 0.5
    D_TH = 0.5
    D_S = 0.5

    # Paso 4-6: Calcular SDk, detectar los puntos de intersección y definir la función de membresía difusa
    for mk in np.arange(0.1, 1.1, 0.1):
        SDk = mk * (SDmax - SD1) + SD1
        
        # Paso 5: Detectar los puntos de intersección entre SDk y los valores de SD
        differences = [abs(SD - SDk) for SD in SD_values]
        min_diff_index = np.argmin(differences)
        cutoff_point1 = gmin + min_diff_index
        
        if min_diff_index + 1 < len(SD_values):
            cutoff_point2 = gmin + min_diff_index + 1
        else:
            cutoff_point2 = gmax
        
        # Paso 6: Definir la función de membresía difusa
        Cp0 = gmin
        Cp3 = gmax
        Cp1 = cutoff_point1
        Cp2 = cutoff_point2
        
        membership_matrix.append([Cp0, Cp1, Cp2, Cp3])
        
        # Paso 7: Calcular RBEMk para encontrar el mejor valor de mk
        RBEMk = calculate_RBEM(Cp0, Cp1, Cp2, Cp3, psi_L, psi_CO, psi_TH, psi_S, D_L, D_CO, D_TH, D_S)
        
        if RBEMk < best_RBEMk:
            best_RBEMk = RBEMk
            best_mk = mk
    
    # Paso adicional: Mapeo de la intensidad de la imagen al dominio difuso y clasificación
    fuzzy_membership_functions = []
    high_memberships = []
    low_memberships = []
    
    for Cpa_minus_1, Cpa in zip([Cp0, Cp1, Cp2], [Cp1, Cp2, Cp3]):
        fuzzy_image = mapping(I, Cpa_minus_1, Cpa)
        high_membership, low_membership = classify_fuzzy_values(fuzzy_image)
        fuzzy_membership_functions.append(fuzzy_image)
        high_memberships.append(high_membership)
        low_memberships.append(low_membership)
    
    return membership_matrix, best_mk, best_RBEMk, fuzzy_membership_functions, high_memberships, low_memberships

# Supongamos que 'I' es la imagen de entrada y gmin, gmax son conocidos
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/IMG_6830.jpeg")

I = cv2.imread(filename)
gmin=
gmax=
membership_matrix, best_mk, best_RBEMk, fuzzy_membership_functions, high_memberships, low_memberships = main(I, gmin, gmax)

# membership_matrix contiene las filas de parámetros de funciones de membresía difusa
# best_mk es el mejor valor de mk encontrado
# best_RBEMk es el valor de RBEMk asociado al mejor mk
# fuzzy_membership_functions contiene las imágenes mapeadas al dominio difuso
# high_memberships y low_memberships contienen las clasificaciones de alta y baja membresía

# Definir los parámetros de las funciones de membresía
Cp0 = 0
Cp1 = 85
Cp2 = 170
Cp3 = 255

# Crear un rango de valores para las funciones de membresía
x = np.linspace(0, 255, 256)

# Definir las funciones de membresía trapezoidales y triangulares con skfuzzy
mu_Cp0 = fuzz.trimf(x, [0, 0, Cp1])
mu_Cp1 = fuzz.trapmf(x, [0, 0, Cp1, Cp2])
mu_Cp2 = fuzz.trapmf(x, [Cp1, Cp1, Cp2, Cp3])

# Graficar las funciones de membresía
plt.figure(figsize=(10, 6))
plt.plot(x, mu_Cp0, label='$\mu_{Cp0}$', color='blue')
plt.plot(x, mu_Cp1, label='$\mu_{Cp1}$', color='green')
plt.plot(x, mu_Cp2, label='$\mu_{Cp2}$', color='red')

plt.title('Funciones de Membresía Fuzzy (skfuzzy)')
plt.xlabel('Intensidad de Grises')
plt.ylabel('Membresía')
plt.legend()
plt.grid(True)
plt.show()

