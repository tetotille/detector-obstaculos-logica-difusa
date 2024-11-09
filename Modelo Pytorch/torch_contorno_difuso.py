import torch
import os
import cv2
import numpy as np

def triangular(x, abc):
    """
    Calcula la función de pertenencia triangular.

    Parameters
    ----------
    x : torch.Tensor
        Tensor de entrada para el cual se calculará la función de pertenencia.
    abc : list or tuple
        Debe contener exactamente tres elementos [a, b, c] que definen la función triangular.
    
    Returns
    -------
    torch.Tensor
        Tensor con los valores de la función de pertenencia triangular.
    """
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = torch.tensor(abc, dtype=torch.float32)  # Desempaquetar los valores de abc
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = torch.zeros_like(x, dtype=torch.float32)

    # Lado izquierdo
    mask_left = (x >= a) & (x <= b)
    y = torch.where(mask_left, (x - a) / (b - a), y)

    # Lado derecho
    mask_right = (x > b) & (x <= c)
    y = torch.where(mask_right, (c - x) / (c - b), y)

    # Punto máximo
    mask_peak = (x == b)
    y = torch.where(mask_peak, torch.float32(1.0), y)

    return y

def defuzzify_centroid(rules, universo):
    """
    Desdifumina las salidas de un conjunto de reglas usando el método del centroide.

    Parameters
    ----------
    rules : list of tuples
        Lista de reglas donde cada regla es un par (final_membership, output_function).
    universo : list
        Lista que define el universo de discurso.

    Returns
    -------
    torch.Tensor
        Tensor con los valores desdifuminados.
    """
    if rules:
        numerator = torch.zeros_like(rules[0][0])  # Mismo tamaño que final_membership
        denominator = torch.zeros_like(rules[0][0])
        
        for final_membership, output_function in rules:
            output_values = output_function(final_membership)  # Asegurar que las dimensiones coincidan
            numerator += final_membership * output_values
            denominator += final_membership
        
        # Evitar división por cero: si el denominador es 0, retorna 0 para esos píxeles
        result = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(denominator))
        
        return result
    
    else:
        # Retorna un tensor de ceros si no hay reglas activadas
        return torch.zeros_like(universo[0])

import torch

def triangular(x, abc):
    """
    Calcula la función de membresía triangular.

    Parameters
    ----------
    x : torch.Tensor
        Tensor de entrada.
    abc : list
        Lista con tres elementos que definen los parámetros de la función triangular.

    Returns
    -------
    torch.Tensor
        Valores de membresía.
    """
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = torch.tensor(abc, dtype=torch.float32)  # Desempaquetar los valores de abc
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = torch.zeros_like(x, dtype=torch.float32)

    # Lado izquierdo
    mask_left = (x >= a) & (x <= b)
    y = torch.where(mask_left, (x - a) / (b - a), y)

    # Lado derecho
    mask_right = (x > b) & (x <= c)
    y = torch.where(mask_right, (c - x) / (c - b), y)

    # Punto máximo
    mask_peak = (x == b)
    y = torch.where(mask_peak, torch.float32(1.0), y)

    return y

def define_membership_functions(image):
    """
    Define funciones de membresía para diferentes categorías de píxeles en una imagen.

    Parameters
    ----------
    image : torch.Tensor
        Tensor que representa la imagen de entrada.

    Returns
    -------
    list, dict
        Una lista de antecedentes con funciones de membresía y un diccionario de funciones de membresía para el borde.
    """
    min_pixel = torch.min(image)
    max_pixel = torch.max(image)

    # Crear funciones de membresía
    low_membership = lambda x: triangular(x, [0, 0, 0.5])
    medium_membership = lambda x: triangular(x, [max_pixel / 3, (min_pixel + max_pixel) / 2, max_pixel * 2 / 3])
    high_membership = lambda x: triangular(x, [0.5, 1.0, 1.0])

    antecedents = [{'low': low_membership, 'medium': medium_membership, 'high': high_membership} for _ in range(9)]

    edge = {
        'low': low_membership,
        'high': high_membership,
        'yes': lambda x: torch.full_like(x, torch.float32(0.5))  # Definición de 'yes' como un singleton
    }

    return antecedents, edge

import torch

def define_rules(antecedents, edge, neighbor_values, universo, output_file="reglas_activadas.txt"):
    """
    Define reglas difusas basadas en los antecedentes y las funciones de membresía.

    Parameters
    ----------
    antecedents : list
        Lista de funciones de membresía para los antecedentes.
    edge : dict
        Diccionario que contiene funciones de membresía para bordes.
    neighbor_values : torch.Tensor
        Valores de vecinos (pueden ser utilizados para reglas adicionales).
    universo : any
        Representación del universo de discurso (no utilizado directamente en este contexto).
    output_file : str, optional
        Nombre del archivo para guardar las reglas activadas. Default es 'reglas_activadas.txt'.

    Returns
    -------
    torch.Tensor
        Valores crisp finales a partir de la defuzzificación de las reglas activadas.
    """
    # Definir las reglas en formato de listas de índices para high, medium, low y la salida
    rule_sets = [
        ([0, 1, 3, 4, 6, 7], [], [2, 5, 8], edge['yes']),  # Ejemplo de regla 1
        ([0, 1, 2, 3, 4, 5], [], [6, 7, 8], edge['yes']),  # Ejemplo de regla 2
        ([3, 4, 5, 6, 7, 8], [], [0, 1, 2], edge['yes']),  # Ejemplo de regla 3
        ([1, 2, 4, 5, 7, 8], [], [0, 4, 6], edge['yes']),  # Ejemplo de regla 4
        ([2, 5, 6, 7, 8], [], [0, 1, 3, 4], edge['yes']),  # Ejemplo de regla 5
        ([0, 1, 2, 5, 8], [], [3, 4, 6, 7], edge['yes']),  # Ejemplo de regla 6
        ([0, 3, 6, 7, 8], [], [1, 2, 4, 5], edge['yes']),  # Ejemplo de regla 7
        ([0, 1, 2, 3, 6], [], [4, 5, 7, 8], edge['yes']),  # Ejemplo de regla 8
        ([5, 7, 8], [], [0, 1, 2, 3, 4, 6], edge['yes']),  # Ejemplo de regla 9
        ([3, 6, 7], [], [0, 1, 2, 4, 5, 8], edge['yes']),  # Ejemplo de regla 10
        ([0, 1, 3], [], [2, 4, 5, 6, 7, 8], edge['yes']),  # Ejemplo de regla 11
        ([1, 2, 5], [], [0, 3, 4, 6, 7, 8], edge['yes']),  # Ejemplo de regla 12
        ([6, 7, 8], [], [0, 1, 2, 3, 4, 5], edge['yes']),  # Ejemplo de regla 13
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']),  # Ejemplo de regla 14
        ([0, 1, 2], [], [3, 4, 5, 6, 7, 8], edge['yes']),  # Ejemplo de regla 15
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']),  # Ejemplo de regla 16
    ]

    # Convertir a matrices de reglas
    indices_high = [rule[0] for rule in rule_sets]
    indices_medium = [rule[1] for rule in rule_sets]
    indices_low = [rule[2] for rule in rule_sets]
    outputs = [rule[3] for rule in rule_sets]

    # Expandir neighbor_values para que cada conjunto corresponda a una regla
    expanded_neighbor_values = neighbor_values.unsqueeze(1).repeat(1, len(rule_sets), 1, 1)

    # Calcular memberships para cada conjunto de reglas
    high_memberships = torch.ones_like(expanded_neighbor_values[0])
    medium_memberships = torch.ones_like(expanded_neighbor_values[0])
    low_memberships = torch.ones_like(expanded_neighbor_values[0])

    for i, indices in enumerate(indices_high):
        for idx in indices:
            high_memberships[i] = torch.min(high_memberships[i], antecedents[idx]['high'](expanded_neighbor_values[:, i]))

    for i, indices in enumerate(indices_medium):
        for idx in indices:
            medium_memberships[i] = torch.min(medium_memberships[i], antecedents[idx]['medium'](expanded_neighbor_values[:, i]))

    for i, indices in enumerate(indices_low):
        for idx in indices:
            low_memberships[i] = torch.min(low_memberships[i], antecedents[idx]['low'](expanded_neighbor_values[:, i]))

    # Calcular la membresía mínima entre high, medium y low para cada regla
    final_memberships = torch.min(torch.min(high_memberships, medium_memberships), low_memberships)

    # Evaluar cuáles reglas se activan y guardar los resultados para la defuzzificación
    rules = []
    for i in range(len(rule_sets)):
        activation = final_memberships[i] > 0.4

        if torch.any(activation):
            # Guardar la membresía final y el valor de salida correspondiente
            rules.append((final_memberships[i], outputs[i]))

    # Procesar las reglas activadas y realizar la defuzzificación
    final_crisp_values = defuzzify_centroid(rules, universo)

    # Retornar la salida defuzzificada
    return final_crisp_values


def apply_fuzzy_rules_to_image(fuzzy_image, antecedents, edge):
    rows, cols = fuzzy_image.shape
    edge_image = torch.zeros((rows, cols), dtype=torch.float32)

    # Definir las posiciones de los vecinos
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1), 
        (0, -1),  (0, 0),  (0, 1),  
        (1, -1),  (1, 0),  (1, 1)   
    ]

    # Crear arrays que representen los índices de los vecinos
    di = torch.tensor([n[0] for n in neighbors])
    dj = torch.tensor([n[1] for n in neighbors])

    # Expandir la imagen para calcular los vecinos de manera eficiente
    expanded_image = torch.nn.functional.pad(fuzzy_image, pad=(1, 1, 1, 1), mode='constant', value=0)

    # Generar las coordenadas i y j para el área válida, excluyendo los bordes
    i_coords, j_coords = torch.meshgrid(torch.arange(1, rows - 1), torch.arange(1, cols - 1), indexing='ij')

    # Calcular los valores de los vecinos para todos los píxeles en paralelo
    neighbor_values = torch.stack([expanded_image[i_coords + di[n], j_coords + dj[n]] for n in range(9)])

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

        except KeyError as e:
            print(f"Error: la clave {str(e)} no se encontró en antecedents para el vecino {idx}.")
            # Rellena con ceros en caso de que no se encuentre la clave, para evitar que falle el proceso
            membership_values['low'] = torch.zeros_like(neighbor_values[idx])
            membership_values['medium'] = torch.zeros_like(neighbor_values[idx])
            membership_values['high'] = torch.zeros_like(neighbor_values[idx])
            membership_values_list.append(membership_values)  
        
        membership_values = membership_values_list[idx]
        
        # Calcular los valores 'high', 'medium', y 'low'
        # Comenzamos por asignar el valor de `high_values` si la condición para `high` es verdadera
        final_values = torch.where((membership_values['high'] > membership_values['medium']) & 
                                   (membership_values['high'] > membership_values['low']),
                                   torch.tensor(1.0, dtype=torch.float32),
                                   fuzzy_image[i_coords + di[idx], j_coords + dj[idx]])

        # Luego, si la condición para `medium` es verdadera y `high` no lo es, usamos `medium_values`
        final_values = torch.where((membership_values['medium'] > membership_values['high']) & 
                                   (membership_values['medium'] > membership_values['low']),
                                   fuzzy_image[i_coords + di[idx], j_coords + dj[idx]],
                                   final_values)

        # Finalmente, si la condición para `low` es verdadera y `high` y `medium` no lo son, usamos `low_values`
        final_values = torch.where((membership_values['low'] > membership_values['high']) & 
                                   (membership_values['low'] > membership_values['medium']),
                                   fuzzy_image[i_coords + di[idx], j_coords + dj[idx]],
                                   final_values)

        # Almacenar los valores finales en la imagen de borde
        edge_image[i_coords, j_coords] = final_values

    return edge_image


def load_image(image):
    # Convertir la imagen de BGR a escala de grises manualmente usando PyTorch
    image = torch.tensor(image)  # Convertir la imagen a un tensor de PyTorch
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale_image

import torch
import os

def process_image(image):
    # Cargar la imagen y convertirla a escala de grises
    grayscale_image = load_image(image)  # Suponiendo que esta función ya está en PyTorch
    min_val = torch.min(grayscale_image)
    max_val = torch.max(grayscale_image)

    if max_val > min_val:  # Para evitar división por cero
        grayscale_image = (grayscale_image - min_val) / (max_val - min_val) * 255.0
    else:
        raise ValueError("Error: min_val es igual a max_val, no se puede normalizar.")
        
    # Normalizar la imagen entre 0 y 1 para procesamiento difuso
    fuzzy_image = grayscale_image.float() / 255.0  # Asegurarse de que sea tipo float y normalizar

    # Definir las funciones de membresía
    antecedents, edge = define_membership_functions(fuzzy_image)  # Asegúrate de que esta función también esté adaptada a PyTorch

    # Aplicar las reglas difusas a la imagen
    edge_image = apply_fuzzy_rules_to_image(fuzzy_image, antecedents, edge)  # Debes asegurarte de que esta función esté adaptada a PyTorch

    # Convertir la imagen procesada a formato uint8
    edge_image_uint8 = torch.clamp(edge_image * 255, 0, 255).byte()  # Clamped y convertida a uint8

    abs_path = os.path.join(os.getcwd(), "imagen_umbral.png")
    #make_contours_white(edge_image_uint8.numpy(), abs_path)  # Asegúrate de que esta función acepte un array de NumPy

    # Retornar la imagen procesada
    return edge_image_uint8
