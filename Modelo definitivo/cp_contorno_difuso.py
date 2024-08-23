import cupy as cp
import numpy as np

# Función de membresía triangular
def triangular(x, abc):
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = cp.array(abc, dtype=cp.float32)  # Desempaquetar los valores de abc
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = cp.zeros_like(x, dtype=cp.float32)

    # Lado izquierdo
    mask_left = (x >= a) & (x <= b)
    y = cp.where(mask_left, (x - a) / (b - a), y)

    # Lado derecho
    mask_right = (x > b) & (x <= c)
    y = cp.where(mask_right, (c - x) / (c - b), y)

    # Punto máximo
    mask_peak = (x == b)
    y = cp.where(mask_peak, cp.float32(1.0), y)

    return y

# Función para escribir los valores de la imagen en un archivo de texto
def write_fuzzy_image_to_file(image, filename="fuzzy_image_values.txt"):
    with open(filename, "w") as f:
        f.write("Valores de fuzzy_image después de la actualización:\n")
        f.write(f"{image.get()}\n")

# Guardar los valores de fuzzy_image en un archivo de texto


def write_membership_values_to_file(membership_values_list, filename="membership_values.txt"):
    with open(filename, "w") as f:
        for idx, membership_values in enumerate(membership_values_list):
            f.write(f"Vecino {idx}:\n")
            for label in ['low', 'medium', 'high']:
                f.write(f"Valores para {label}:\n")
                values = membership_values[label].get()  # Convertir a NumPy
                f.write(f"{values}\n\n")


# Función de defuzzificación por centroide
def defuzzify_centroid(rules, universo):
    if rules:
        numerator = cp.zeros_like(rules[0][0])  # Mismo tamaño que final_membership
        denominator = cp.zeros_like(rules[0][0])
        
        for final_membership, output_function in rules:
            output_values = output_function(final_membership)  # Asegurar que las dimensiones coincidan
            numerator += final_membership * output_values
            denominator += final_membership
        
        # Evitar división por cero: si el denominador es 0, retorna 0 para esos píxeles
        result = cp.where(denominator != 0, numerator / denominator, cp.zeros_like(denominator))
        
        return result
    
    else:
        # Retorna un array de ceros si no hay reglas activadas
        return cp.zeros_like(universo[0])

# Definición de funciones de membresía
def define_membership_functions(image):
    min_pixel = cp.min(image)
    max_pixel = cp.max(image)

    # Crear funciones de membresía
    low_membership = lambda x: triangular(x, [0, 0, 0.5])
    medium_membership = lambda x: triangular(x, [max_pixel / 3, (min_pixel + max_pixel) / 2, max_pixel * 2 / 3])
    high_membership = lambda x: triangular(x, [0.5, 1.0, 1.0])

    antecedents = [{'low': low_membership, 'medium': medium_membership, 'high': high_membership} for _ in range(9)]

    # Supongamos que tienes un valor Python nativo

    edge = {
        'low': low_membership,
        'high': high_membership,
        'yes': lambda x: cp.full_like(x, cp.float32(0.5))  # Definición de 'yes' como un singleton
    }

    return antecedents, edge

# Definición de reglas difusas
def define_rules(antecedents, edge, neighbor_values, universo, output_file="reglas_activadas.txt"):

    # Definir las reglas en formato de listas de índices para high, medium, low y la salida
    rule_sets = [
        ([0, 1, 3, 4, 6, 7], [], [2, 5, 8], edge['yes']),  # Ejemplo de regla 1
        ([0, 1, 2, 3, 4, 5], [], [6, 7, 8], edge['yes']),  # Ejemplo de regla 2
        ([3, 4, 5, 6, 7, 8], [], [0, 1, 2], edge['yes']),  # Ejemplo de regla 3
        ([1, 2, 4, 5, 7, 8], [], [0, 4, 6], edge['yes']), #4
        ([2, 5, 6, 7, 8], [], [0, 1, 3, 4], edge['yes']), #5
        ([0, 1, 2, 5, 8], [], [3, 4, 6, 7], edge['yes']), #6
        ([0, 3, 6, 7, 8], [], [1, 2, 4, 5], edge['yes']), #7
        ([0, 1, 2, 3, 6], [], [4, 5, 7, 8], edge['yes']), #8
        ([5, 7, 8], [], [0, 1, 2, 3, 4, 6], edge['yes']), #9
        ([3, 6, 7], [], [0, 1, 2, 4, 5, 8], edge['yes']), #10
        ([0, 1, 3], [], [2, 4, 5, 6, 7, 8], edge['yes']), #11
        ([1, 2, 5], [], [0, 3, 4, 6, 7, 8], edge['yes']), #12
    #create_rule([], [], [0, 3, 4, 6, 1, 2, 5, 7, 8], edge['low']) #13
    #create_rule([0, 3, 4, 6, 1, 2, 5, 7, 8], [], [], edge['low']) #14
        ([6, 7, 8], [], [0, 1, 2, 3, 4, 5], edge['yes']), #15
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']), #16
        ([0, 1, 2], [], [3, 4, 5, 6, 7, 8], edge['yes']), #17
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']), #18
        ([2, 5, 8], [], [0, 1, 3, 4, 6, 7], edge['yes']), #19
        ([3, 4, 6, 7], [], [0, 1, 2, 5, 8], edge['yes']), #20
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']), #21
        ([4, 5, 7, 8], [], [0, 1, 2, 3, 6], edge['yes']), #22
        ([0, 1, 3, 4], [], [2, 5, 6, 7, 8], edge['yes']), #23
        ([0, 1, 2, 4, 5, 8], [], [3, 6, 7], edge['yes']), #24
        ([2, 4, 5, 6, 7, 8], [], [0, 1, 3], edge['yes']), #25
        ([0, 1, 2, 3, 4, 6], [], [5, 7, 8], edge['yes']), #26
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']), #27
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']), #28
    #create_rule([1, 4, 7], [], [0, 2, 3, 5, 6, 8], edge['yes']) #29
    #create_rule([0, 2, 3, 5, 6, 8], [], [1, 4, 7], edge['yes']) #30
        ([0, 1, 2, 3, 5], [4, 6, 7, 8], [], edge['yes']), #31
        ([0, 1, 3, 6, 7], [2, 4, 5, 8], [], edge['yes']), #32
        ([3, 5, 6, 7, 8], [0, 1, 2, 4], [], edge['yes']), #33
        ([0, 3, 4, 6, 7, 8], [1, 2, 5], [], edge['yes']), #34
        ([0, 3, 4, 5], [1, 2, 5, 7, 8], [], edge['yes']), #35
        ([4, 6, 7, 8], [0, 1, 2, 3, 5], [], edge['yes']), #36
        ([2, 4, 5, 8], [0, 1, 3, 6, 7], [], edge['yes']), #37
        ([0, 1, 2, 4], [3, 5, 6, 7, 8], [], edge['yes']), #38
        ([0, 3, 6, 7], [1, 2, 4, 5, 8], [], edge['yes']), #39
        ([3, 6, 7, 8], [0, 1, 2, 4, 5], [], edge['yes']), #40
        ([5, 6, 7, 8], [0, 1, 2, 3, 4], [], edge['yes']), #41
        ([0, 1, 3, 4, 6], [2, 5, 7, 8], [], edge['yes']), #42
        ([3, 5, 6, 7, 8], [0, 1, 2, 4], [], edge['yes']), #43
        ([0, 1, 3, 6], [2, 4, 5, 7, 8], [], edge['yes']), #44
        ([1, 2, 4, 5, 8], [0, 3, 6, 7], [], edge['yes']), #45
        ([2, 4, 5, 7, 8], [0, 1, 3, 6], [], edge['yes']), #46
        ([4, 5, 6, 7, 8], [0, 1, 2, 3], [], edge['yes']), #47
        ([3, 4, 6, 7, 8], [0, 1, 2, 5], [], edge['yes']), #48
        ([0, 3, 4, 6, 7], [1, 2, 5, 8], [], edge['yes']), #49
        ([0, 1, 3, 4, 6], [2, 5, 7, 8], [], edge['yes']), #50
        ([0, 1, 2, 4, 5], [3, 6, 7, 8], [], edge['yes']), #51
        ([0, 1, 2, 3, 4], [5, 6, 7, 8], [], edge['yes']), #52

        ([0, 1, 3, 4, 6, 7], [2, 5, 8], [], edge['yes']),  # Ejemplo de regla 1
        ([0, 1, 2, 3, 4, 5], [6, 7, 8], [], edge['yes']),  # Ejemplo de regla 2
        ([3, 4, 5, 6, 7, 8], [0, 1, 2], [], edge['yes']),  # Ejemplo de regla 3
        ([1, 2, 4, 5, 7, 8], [0, 4, 6], [], edge['yes']), #4
        ([2, 5, 6, 7, 8], [0, 1, 3, 4], [], edge['yes']), #5
        ([0, 1, 2, 5, 8], [3, 4, 6, 7], [], edge['yes']), #6
        ([0, 3, 6, 7, 8], [1, 2, 4, 5], [], edge['yes']), #7
        ([0, 1, 2, 3, 6], [4, 5, 7, 8], [], edge['yes']), #8
        ([5, 7, 8], [0, 1, 2, 3, 4, 6], [], edge['yes']), #9
        ([3, 6, 7], [0, 1, 2, 4, 5, 8], [], edge['yes']), #10
        ([0, 1, 3], [2, 4, 5, 6, 7, 8], [], edge['yes']), #11
        ([1, 2, 5], [0, 3, 4, 6, 7, 8], [], edge['yes']), #12
    #create_rule([], [], [0, 3, 4, 6, 1, 2, 5, 7, 8], edge['low']) #13
    #create_rule([0, 3, 4, 6, 1, 2, 5, 7, 8], [], [], edge['low']) #14
        ([6, 7, 8], [0, 1, 2, 3, 4, 5], [], edge['yes']), #15
        ([0, 3, 6], [1, 2, 4, 5, 7, 8], [], edge['yes']), #16
        ([0, 1, 2], [3, 4, 5, 6, 7, 8], [], edge['yes']), #17
        ([3, 4, 5], [0, 1, 2, 6, 7, 8], [], edge['yes']), #18
        ([2, 5, 8], [0, 1, 3, 4, 6, 7], [], edge['yes']), #19
        ([3, 4, 6, 7], [0, 1, 2, 5, 8], [], edge['yes']), #20
        ([0, 3, 6], [1, 2, 4, 5, 7, 8], [], edge['yes']), #21
        ([4, 5, 7, 8], [0, 1, 2, 3, 6], [], edge['yes']), #22
        ([0, 1, 3, 4], [2, 5, 6, 7, 8], [], edge['yes']), #23
        ([0, 1, 2, 4, 5, 8], [3, 6, 7], [], edge['yes']), #24
        ([2, 4, 5, 6, 7, 8], [0, 1, 3], [], edge['yes']), #25
        ([0, 1, 2, 3, 4, 6], [5, 7, 8], [], edge['yes']), #26
        ([3, 4, 5], [0, 1, 2, 6, 7, 8], [], edge['yes']), #27
        ([3, 5, 6, 7, 8], [0, 1, 2, 4], [], edge['yes']), #28
    #create_rule([1, 4, 7], [], [0, 2, 3, 5, 6, 8], edge['yes']) #29
    #create_rule([0, 2, 3, 5, 6, 8], [], [1, 4, 7], edge['yes']) #30
        ([0, 1, 2, 3, 5], [], [4, 6, 7, 8], edge['yes']), #31
        ([0, 1, 3, 6, 7], [], [2, 4, 5, 8], edge['yes']), #32
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']), #33
        ([0, 3, 4, 6, 7, 8], [], [1, 2, 5], edge['yes']), #34
        ([0, 3, 4, 5], [], [1, 2, 5, 7, 8], edge['yes']), #35
        ([4, 6, 7, 8], [], [0, 1, 2, 3, 5], edge['yes']), #36
        ([2, 4, 5, 8], [], [0, 1, 3, 6, 7], edge['yes']), #37
        ([0, 1, 2, 4], [], [3, 5, 6, 7, 8], edge['yes']), #38
        ([0, 3, 6, 7], [], [1, 2, 4, 5, 8], edge['yes']), #39
        ([3, 6, 7, 8], [], [0, 1, 2, 4, 5], edge['yes']), #40
        ([5, 6, 7, 8], [], [0, 1, 2, 3, 4], edge['yes']), #41
        ([0, 1, 3, 4, 6], [], [2, 5, 7, 8], edge['yes']), #42
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']), #43
        ([0, 1, 3, 6], [], [2, 4, 5, 7, 8], edge['yes']), #44
        ([1, 2, 4, 5, 8], [], [0, 3, 6, 7], edge['yes']), #45
        ([2, 4, 5, 7, 8], [], [0, 1, 3, 6], edge['yes']), #46
        ([4, 5, 6, 7, 8], [], [0, 1, 2, 3], edge['yes']), #47
        ([3, 4, 6, 7, 8], [], [0, 1, 2, 5], edge['yes']), #48
        ([0, 3, 4, 6, 7], [], [1, 2, 5, 8], edge['yes']), #49
        ([0, 1, 3, 4, 6], [], [2, 5, 7, 8], edge['yes']), #50
        ([0, 1, 2, 4, 5], [], [3, 6, 7, 8], edge['yes']), #51
        ([0, 1, 2, 3, 4], [], [5, 6, 7, 8], edge['yes']), #52

        ([], [0, 1, 3, 4, 6, 7], [2, 5, 8], edge['yes']),  # Ejemplo de regla 1
        ([], [0, 1, 2, 3, 4, 5], [6, 7, 8], edge['yes']),  # Ejemplo de regla 2
        ([], [3, 4, 5, 6, 7, 8], [0, 1, 2], edge['yes']),  # Ejemplo de regla 3
        ([], [1, 2, 4, 5, 7, 8], [0, 4, 6], edge['yes']), #4
        ([], [2, 5, 6, 7, 8], [0, 1, 3, 4], edge['yes']), #5
        ([], [0, 1, 2, 5, 8], [3, 4, 6, 7], edge['yes']), #6
        ([], [0, 3, 6, 7, 8], [1, 2, 4, 5], edge['yes']), #7
        ([], [0, 1, 2, 3, 6], [4, 5, 7, 8], edge['yes']), #8
        ([], [5, 7, 8], [0, 1, 2, 3, 4, 6], edge['yes']), #9
        ([], [3, 6, 7], [0, 1, 2, 4, 5, 8], edge['yes']), #10
        ([], [0, 1, 3], [2, 4, 5, 6, 7, 8], edge['yes']), #11
        ([], [1, 2, 5], [0, 3, 4, 6, 7, 8], edge['yes']), #12
    #create_rule([], [], [0, 3, 4, 6, 1, 2, 5, 7, 8], edge['low']) #13
    #create_rule([0, 3, 4, 6, 1, 2, 5, 7, 8], [], [], edge['low']) #14
        ([], [6, 7, 8], [0, 1, 2, 3, 4, 5], edge['yes']), #15
        ([], [0, 3, 6], [1, 2, 4, 5, 7, 8], edge['yes']), #16
        ([], [0, 1, 2], [3, 4, 5, 6, 7, 8], edge['yes']), #17
        ([], [3, 4, 5], [0, 1, 2, 6, 7, 8], edge['yes']), #18
        ([], [2, 5, 8], [0, 1, 3, 4, 6, 7], edge['yes']), #19
        ([], [3, 4, 6, 7], [0, 1, 2, 5, 8], edge['yes']), #20
        ([], [0, 3, 6], [1, 2, 4, 5, 7, 8], edge['yes']), #21
        ([], [4, 5, 7, 8], [0, 1, 2, 3, 6], edge['yes']), #22
        ([], [0, 1, 3, 4], [2, 5, 6, 7, 8], edge['yes']), #23
        ([], [0, 1, 2, 4, 5, 8], [3, 6, 7], edge['yes']), #24
        ([], [2, 4, 5, 6, 7, 8], [0, 1, 3], edge['yes']), #25
        ([], [0, 1, 2, 3, 4, 6], [5, 7, 8], edge['yes']), #26
        ([], [3, 4, 5], [0, 1, 2, 6, 7, 8], edge['yes']), #27
        ([], [3, 5, 6, 7, 8], [0, 1, 2, 4], edge['yes']), #28
    #create_rule([1, 4, 7], [], [0, 2, 3, 5, 6, 8], edge['yes']) #29
    #create_rule([0, 2, 3, 5, 6, 8], [], [1, 4, 7], edge['yes']) #30
        ([], [0, 1, 2, 3, 5], [4, 6, 7, 8], edge['yes']), #31
        ([], [0, 1, 3, 6, 7], [2, 4, 5, 8], edge['yes']), #32
        ([], [3, 5, 6, 7, 8], [0, 1, 2, 4], edge['yes']), #33
        ([], [0, 3, 4, 6, 7, 8], [1, 2, 5], edge['yes']), #34
        ([], [0, 3, 4, 5], [1, 2, 5, 7, 8], edge['yes']), #35
        ([], [4, 6, 7, 8], [0, 1, 2, 3, 5], edge['yes']), #36
        ([], [2, 4, 5, 8], [0, 1, 3, 6, 7], edge['yes']), #37
        ([], [0, 1, 2, 4], [3, 5, 6, 7, 8], edge['yes']), #38
        ([], [0, 3, 6, 7], [1, 2, 4, 5, 8], edge['yes']), #39
        ([], [3, 6, 7, 8], [0, 1, 2, 4, 5], edge['yes']), #40
        ([], [5, 6, 7, 8], [0, 1, 2, 3, 4], edge['yes']), #41
        ([], [0, 1, 3, 4, 6], [2, 5, 7, 8], edge['yes']), #42
        ([], [3, 5, 6, 7, 8], [0, 1, 2, 4], edge['yes']), #43
        ([], [0, 1, 3, 6], [2, 4, 5, 7, 8], edge['yes']), #44
        ([], [1, 2, 4, 5, 8], [0, 3, 6, 7], edge['yes']), #45
        ([], [2, 4, 5, 7, 8], [0, 1, 3, 6], edge['yes']), #46
        ([], [4, 5, 6, 7, 8], [0, 1, 2, 3], edge['yes']), #47
        ([], [3, 4, 6, 7, 8], [0, 1, 2, 5], edge['yes']), #48
        ([], [0, 3, 4, 6, 7], [1, 2, 5, 8], edge['yes']), #49
        ([], [0, 1, 3, 4, 6], [2, 5, 7, 8], edge['yes']), #50
        ([], [0, 1, 2, 4, 5], [3, 6, 7, 8], edge['yes']), #51
        ([], [0, 1, 2, 3, 4], [5, 6, 7, 8], edge['yes']), #52
        # Agrega más reglas según sea necesario...
    ]

    # Convertir a matriz de reglas
    indices_high = [rule[0] for rule in rule_sets]
    indices_medium = [rule[1] for rule in rule_sets]
    indices_low = [rule[2] for rule in rule_sets]
    outputs = [rule[3] for rule in rule_sets]

    # Expandir neighbor_values para que cada conjunto corresponda a una regla
    expanded_neighbor_values = cp.repeat(neighbor_values[:, cp.newaxis, :, :], len(rule_sets), axis=1)

    # Calcular memberships para cada conjunto de reglas
    high_memberships = cp.ones_like(expanded_neighbor_values[0])
    medium_memberships = cp.ones_like(expanded_neighbor_values[0])
    low_memberships = cp.ones_like(expanded_neighbor_values[0])

    for i, indices in enumerate(indices_high):
        for idx in indices:
            high_memberships[i] = cp.minimum(high_memberships[i], antecedents[idx]['high'](expanded_neighbor_values[idx, i]))

    for i, indices in enumerate(indices_medium):
        for idx in indices:
            medium_memberships[i] = cp.minimum(medium_memberships[i], antecedents[idx]['medium'](expanded_neighbor_values[idx, i]))

    for i, indices in enumerate(indices_low):
        for idx in indices:
            low_memberships[i] = cp.minimum(low_memberships[i], antecedents[idx]['low'](expanded_neighbor_values[idx, i]))

    # Calcular la membresía mínima entre high, medium y low para cada regla
    final_memberships = cp.minimum(cp.minimum(high_memberships, medium_memberships), low_memberships)

    # Evaluar cuáles reglas se activan y guardar los resultados para la defuzzificación
    rules = []
    for i in range(len(rule_sets)):
        activation = final_memberships[i] > 0.4

        if cp.any(activation):
            with open(output_file, "a") as f:
                f.write(f"Regla {i + 1} activada con final_membership:\n")
                np_final_membership = final_memberships[i].get()  # Convertir a NumPy para imprimir
                np.set_printoptions(precision=10, suppress=False, floatmode='fixed')  # Ajustar precisión decimal
                f.write(np.array2string(np_final_membership, separator=', ') + "\n")
            
            # Guardar la membresía final y el valor de salida correspondiente
            rules.append((final_memberships[i], outputs[i]))

    # Procesar las reglas activadas y realizar la defuzzificación
    final_crisp_values = defuzzify_centroid(rules, universo)
    
    print(final_crisp_values.get())  # Utiliza .get() para obtener el array en formato NumPy desde CuPy
    
    with open(output_file, "a") as f:
        f.write(f"Salida defuzzificada por píxel: {final_crisp_values.get()}\n")

    # Retornar la salida defuzzificada
    return final_crisp_values

# Aplicar reglas difusas a la imagen
"""def print_antecedents(antecedents):
    for idx, antecedent in enumerate(antecedents):
        print(f"Antecedent {idx + 1}:")
        for label in ['low', 'medium', 'high']:
            print(f"  {label} function:")
            # Imprime la función de membresía para cada etiqueta
            function = antecedent[label]
            print(function)"""

# Aplicar reglas difusas a la imagen
def apply_fuzzy_rules_to_image(fuzzy_image, antecedents, edge):
    rows, cols = fuzzy_image.shape
    edge_image = cp.zeros((rows, cols), dtype=cp.float32)

    # Definir las posiciones de los vecinos
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1), 
        (0, -1),  (0, 0),  (0, 1),  
        (1, -1),  (1, 0),  (1, 1)   
    ]

    # Crear arrays que representen los índices de los vecinos
    di = cp.array([n[0] for n in neighbors])
    dj = cp.array([n[1] for n in neighbors])

    # Expandir la imagen para calcular los vecinos de manera eficiente
    expanded_image = cp.pad(fuzzy_image, pad_width=1, mode='constant', constant_values=0)

    # Generar las coordenadas i y j para el área válida, excluyendo los bordes
    i_coords, j_coords = cp.meshgrid(cp.arange(1, rows - 1), cp.arange(1, cols - 1), indexing='ij')

    # Calcular los valores de los vecinos para todos los píxeles en paralelo
    neighbor_values = cp.array([expanded_image[i_coords + di[n], j_coords + dj[n]] for n in range(9)])

    #Convertir a NumPy para visualizar e imprimir

    
    # Inicializar lista para almacenar las membresías calculadas
    membership_values_list = []

    # Calcular los valores de membresía para cada vecino y cada conjunto difuso
    for idx in range(9):
        membership_values = {}
        try:
            for label in ['low', 'medium', 'high']:
                # Aplicar la función de membresía a los valores de los vecinos
                membership_values[label] = antecedents[idx][label](neighbor_values[idx])
            membership_values_list.append(membership_values)
            write_membership_values_to_file(membership_values_list, filename="membership_values.txt")

        except KeyError as e:
            print(f"Error: la clave {str(e)} no se encontró en antecedents para el vecino {idx}.")
            # Rellena con ceros en caso de que no se encuentre la clave, para evitar que falle el proceso
            membership_values['low'] = cp.zeros_like(neighbor_values[idx])
            membership_values['medium'] = cp.zeros_like(neighbor_values[idx])
            membership_values['high'] = cp.zeros_like(neighbor_values[idx])
            membership_values_list.append(membership_values)  
    
        membership_values = membership_values_list[idx]
        # Calcular los valores 'high', 'medium', y 'low'

        # Comenzamos por asignar el valor de `high_values` si la condición para `high` es verdadera
        final_values = cp.where((membership_values['high'] > membership_values['medium']) & 
                                (membership_values['high'] > membership_values['low']), cp.float32(1.0),fuzzy_image[i_coords + di[idx], j_coords + dj[idx]])

        # Luego, si la condición para `medium` es verdadera y `high` no lo es, usamos `medium_values`
        final_values = cp.where((membership_values['medium'] > membership_values['high']) & 
                                (membership_values['medium'] > membership_values['low']), cp.float32(0.5), final_values)

        # Finalmente, si `low` es la condición verdadera, actualizamos `final_values` con `low_values`
        final_values = cp.where((membership_values['low'] > membership_values['medium']) & 
                                (membership_values['low'] > membership_values['high']), cp.float32(0.0), final_values)

        # Actualizar la imagen difusa con el valor final
        fuzzy_image[i_coords + di[idx], j_coords + dj[idx]] = final_values

        height, width = fuzzy_image.shape

        

    write_fuzzy_image_to_file(fuzzy_image, filename="fuzzy_image_values.txt")
    
    # Actualizar neighbor_values con la nueva fuzzy_image
    expanded_image = cp.pad(fuzzy_image, pad_width=1, mode='constant', constant_values=0)
    neighbor_values = cp.array([expanded_image[i_coords + di[n], j_coords + dj[n]] for n in range(9)])
    neighbor_values_numpy = neighbor_values.get()

    #print("Valores de los vecinos:")
    #print(neighbor_values_numpy)
    # Generar 50 índices aleatorios
    height, width = neighbor_values.shape[1], neighbor_values.shape[2]

# Generar 50 índices aleatorios dentro de los límites de neighbor_values
    random_indices = np.random.choice(height * width, 50, replace=False)

    # Convertir los índices planos a coordenadas 2D dentro de los límites de neighbor_values
    random_coords = np.unravel_index(random_indices, (height, width))

    # Obtener los valores de neighbor_values en esas coordenadas
    random_pixel_values = neighbor_values[0, random_coords[0], random_coords[1]].get()

    # Mostrar los 50 valores aleatorios seleccionados
    #print("Valores píxeles random:", random_pixel_values)

    # Imprimir 50 antecedentes de la etiqueta 'high'
    antecedents_high_values = []
    """for i in range(50):
        x, y = random_coords[0][i], random_coords[1][i]
        high_value = antecedents[0]['high'](neighbor_values[0][x, y]).get()  # Cambia el índice [0] si quieres usar otros vecinos
        antecedents_high_values.append(high_value)"""

    # Mostrar los valores de antecedentes 'high'
    #print("Valores de antecedentes 'high':", antecedents_high_values)

    # Mostrar los valores de antecedentes 'high'
    
    # Ahora, recalcular las reglas usando la fuzzy_image modificada y los nuevos neighbor_values
    rules = define_rules(antecedents, edge, neighbor_values, cp.linspace(0, 1, 256))
    # Evaluar reglas y determinar salida
    central_pixel_output = cp.zeros((rows - 2, cols - 2), dtype=cp.float32)
        # Actualizar el valor de salida basado en la regla que se cumple
    central_pixel_output = cp.maximum(central_pixel_output, rules)

    # Asignar el valor final al píxel central
    edge_image[i_coords, j_coords] = cp.where(central_pixel_output > 0, central_pixel_output, 0)

    return edge_image

# Cargar y convertir la imagen a escala de grises
def load_image(image):
    # Convertir la imagen de BGR a escala de grises manualmente usando CuPy
    image = cp.asarray(image)
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale_image

# Redimensionar imagen
def resize_image(image, new_shape):
    # Obtén las dimensiones originales y las nuevas dimensiones
    orig_shape = image.shape
    new_height, new_width = new_shape

    # Crea matrices para las nuevas coordenadas
    y = cp.linspace(0, orig_shape[0] - 1, new_height)
    x = cp.linspace(0, orig_shape[1] - 1, new_width)
    x_grid, y_grid = cp.meshgrid(x, y)

    # Interpolación bilineal
    x0 = cp.floor(x_grid).astype(cp.int32)
    x1 = x0 + 1
    y0 = cp.floor(y_grid).astype(cp.int32)
    y1 = y0 + 1

    x0 = cp.clip(x0, 0, orig_shape[1] - 1)
    x1 = cp.clip(x1, 0, orig_shape[1] - 1)
    y0 = cp.clip(y0, 0, orig_shape[0] - 1)
    y1 = cp.clip(y1, 0, orig_shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x_grid) * (y1 - y_grid)
    wb = (x1 - x_grid) * (y_grid - y0)
    wc = (x_grid - x0) * (y1 - y_grid)
    wd = (x_grid - x0) * (y_grid - y0)

    resized_image = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # Asegurarse de que los valores estén dentro del rango [0, 255]
    resized_image = cp.clip(resized_image, 0, 255)

    return resized_image

def write_image_to_file(image, filename="edge_image_values.txt"):
    with open(filename, "w") as f:
        # Convertir la imagen de CuPy a NumPy para que se pueda escribir en el archivo
        image_np = image.get()
        for row in image_np:
            f.write(" ".join(map(str, row)) + "\n")


# Procesar la imagen
def process_image(image):
    # Cargar la imagen y convertirla a escala de grises
    grayscale_image = load_image(image)
    min_val = cp.min(grayscale_image)
    max_val = cp.max(grayscale_image)
    if max_val > min_val:  # Para evitar división por cero
        grayscale_image = (grayscale_image - min_val) / (max_val - min_val) * 255.0
    else:
        raise ValueError("Error: min_val es igual a max_val, no se puede normalizar.")
        
    # Redimensionar la imagen (opcional, dependiendo de la necesidad)
    # Normalizar la imagen entre 0 y 1 para procesamiento difuso
    fuzzy_image = grayscale_image.astype(cp.float32) / 256.0

    # Definir las funciones de membresía
    antecedents, edge = define_membership_functions(fuzzy_image)
    #print_antecedents(antecedents)

    # Aplicar las reglas difusas a la imagen
    edge_image = apply_fuzzy_rules_to_image(fuzzy_image, antecedents, edge)

    write_image_to_file(edge_image, filename="edge_image_values.txt")

    # Convertir la imagen procesada a formato uint8
    edge_image_uint8 = cp.clip(edge_image * 255, 0, 255).astype(cp.uint8)
    edge_image_uint8_numpy = edge_image_uint8.get()
    cp.get_default_memory_pool().free_all_blocks()
    # Retornar la imagen procesada
    return edge_image_uint8_numpy

# Umbral adaptativo con CuPy
def adaptive_threshold(image, block_size, C):
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)

    if block_size % 2 == 0:
        raise ValueError("block_size debe ser un número impar.")

    mean_filter = cp.ones((block_size, block_size), dtype=cp.float32) / (block_size * block_size)
    mean_image = cp.signal.convolve(image, mean_filter, mode='same')

    thresholded_image = image - mean_image - C
    thresholded_image = cp.where(thresholded_image > 0, 255, 0).astype(cp.uint8)

    return thresholded_image

