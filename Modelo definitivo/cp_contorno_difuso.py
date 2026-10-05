import numpy as np
import os
import cv2

# Usamos numpy en lugar de cupy para CPU
cp = np

# Función de membresía triangular
def triangular(x, abc):
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = cp.array(abc, dtype=cp.float32)
    assert a <= b and b <= c, f'abc requires the three elements a <= b <= c. Got {a}, {b}, {c}'

    y = cp.zeros_like(x, dtype=cp.float32)

    # Lado izquierdo
    if b > a:
        mask_left = (x >= a) & (x <= b)
        y = cp.where(mask_left, (x - a) / (b - a), y)

    # Lado derecho
    if c > b:
        mask_right = (x > b) & (x <= c)
        y = cp.where(mask_right, (c - x) / (c - b), y)

    # Punto máximo
    mask_peak = (x == b)
    y = cp.where(mask_peak, cp.float32(1.0), y)

    return y

# Función de defuzzificación por centroide
def defuzzify_centroid(rules, universo):
    if rules:
        numerator = cp.zeros_like(rules[0][0])
        denominator = cp.zeros_like(rules[0][0])
        
        for final_membership, output_function in rules:
            output_values = output_function(final_membership)
            numerator += final_membership * output_values
            denominator += final_membership
        
        # Evitar división por cero y advertencias de Runtime
        result = cp.zeros_like(numerator)
        cp.divide(numerator, denominator, out=result, where=denominator != 0)
        
        return result
    else:
        return cp.zeros_like(universo)

# Definición de reglas difusas
def define_rules(antecedents, edge, neighbor_values, universo):
    rule_sets = [
        ([0, 1, 3, 4, 6, 7], [], [2, 5, 8], edge['yes']),
        ([0, 1, 2, 3, 4, 5], [], [6, 7, 8], edge['yes']),
        ([3, 4, 5, 6, 7, 8], [], [0, 1, 2], edge['yes']),
        ([1, 2, 4, 5, 7, 8], [], [0, 4, 6], edge['yes']),
        ([2, 5, 6, 7, 8], [], [0, 1, 3, 4], edge['yes']),
        ([0, 1, 2, 5, 8], [], [3, 4, 6, 7], edge['yes']),
        ([0, 3, 6, 7, 8], [], [1, 2, 4, 5], edge['yes']),
        ([0, 1, 2, 3, 6], [], [4, 5, 7, 8], edge['yes']),
        ([5, 7, 8], [], [0, 1, 2, 3, 4, 6], edge['yes']),
        ([3, 6, 7], [], [0, 1, 2, 4, 5, 8], edge['yes']),
        ([0, 1, 3], [], [2, 4, 5, 6, 7, 8], edge['yes']),
        ([1, 2, 5], [], [0, 3, 4, 6, 7, 8], edge['yes']),
        ([6, 7, 8], [], [0, 1, 2, 3, 4, 5], edge['yes']),
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']),
        ([0, 1, 2], [], [3, 4, 5, 6, 7, 8], edge['yes']),
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']),
        ([2, 5, 8], [], [0, 1, 3, 4, 6, 7], edge['yes']),
        ([3, 4, 6, 7], [], [0, 1, 2, 5, 8], edge['yes']),
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']),
        ([4, 5, 7, 8], [], [0, 1, 2, 3, 6], edge['yes']),
        ([0, 1, 3, 4], [], [2, 5, 6, 7, 8], edge['yes']),
        ([0, 1, 2, 4, 5, 8], [], [3, 6, 7], edge['yes']),
        ([2, 4, 5, 6, 7, 8], [], [0, 1, 3], edge['yes']),
        ([0, 1, 2, 3, 4, 6], [], [5, 7, 8], edge['yes']),
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']),
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']),
        ([0, 1, 2, 3, 5], [4, 6, 7, 8], [], edge['yes']),
        ([0, 1, 3, 6, 7], [2, 4, 5, 8], [], edge['yes']),
        ([3, 5, 6, 7, 8], [0, 1, 2, 4], [], edge['yes']),
        ([0, 3, 4, 6, 7, 8], [1, 2, 5], [], edge['yes']),
        ([0, 3, 4, 5], [1, 2, 5, 7, 8], [], edge['yes']),
        ([4, 6, 7, 8], [0, 1, 2, 3, 5], [], edge['yes']),
        ([2, 4, 5, 8], [0, 1, 3, 6, 7], [], edge['yes']),
        ([0, 1, 2, 4], [3, 5, 6, 7, 8], [], edge['yes']),
        ([0, 3, 6, 7], [1, 2, 4, 5, 8], [], edge['yes']),
        ([3, 6, 7, 8], [0, 1, 2, 4, 5], [], edge['yes']),
        ([5, 6, 7, 8], [0, 1, 2, 3, 4], [], edge['yes']),
        ([0, 1, 3, 4, 6], [2, 5, 7, 8], [], edge['yes']),
        ([3, 5, 6, 7, 8], [0, 1, 2, 4], [], edge['yes']),
        ([0, 1, 3, 6], [2, 4, 5, 7, 8], [], edge['yes']),
        ([1, 2, 4, 5, 8], [0, 3, 6, 7], [], edge['yes']),
        ([2, 4, 5, 7, 8], [0, 1, 3, 6], [], edge['yes']),
        ([4, 5, 6, 7, 8], [0, 1, 2, 3], [], edge['yes']),
        ([3, 4, 6, 7, 8], [0, 1, 2, 5], [], edge['yes']),
        ([0, 3, 4, 6, 7], [1, 2, 5, 8], [], edge['yes']),
        ([0, 1, 3, 4, 6], [2, 5, 7, 8], [], edge['yes']),
        ([0, 1, 2, 4, 5], [3, 6, 7, 8], [], edge['yes']),
        ([0, 1, 2, 3, 4], [5, 6, 7, 8], [], edge['yes']),
    ]

    indices_high = [rule[0] for rule in rule_sets]
    indices_medium = [rule[1] for rule in rule_sets]
    indices_low = [rule[2] for rule in rule_sets]
    outputs = [rule[3] for rule in rule_sets]

    expanded_neighbor_values = cp.repeat(neighbor_values[:, cp.newaxis, :, :], len(rule_sets), axis=1)

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

    final_memberships = cp.minimum(cp.minimum(high_memberships, medium_memberships), low_memberships)

    rules = []
    for i in range(len(rule_sets)):
        activation = final_memberships[i] > 0.4
        if cp.any(activation):
            rules.append((final_memberships[i], outputs[i]))

    final_crisp_values = defuzzify_centroid(rules, universo)
    return final_crisp_values

def load_image(image):
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale_image

def process_image(image):
    grayscale_image = load_image(image)
    min_val = float(cp.min(grayscale_image))
    max_val = float(cp.max(grayscale_image))
    
    if max_val > min_val:
        grayscale_image = (grayscale_image - min_val) / (max_val - min_val) * 255.0
    else:
        grayscale_image = cp.zeros_like(grayscale_image)
    
    fuzzy_image = grayscale_image.astype(cp.float32) / 256.0

    min_pix = float(cp.min(fuzzy_image))
    max_pix = float(cp.max(fuzzy_image))

    low_membership = lambda x: triangular(x, [0, 0, 0.5])
    medium_membership = lambda x: triangular(x, [max_pix / 3, (min_pix + max_pix) / 2, max_pix * 2 / 3])
    high_membership = lambda x: triangular(x, [0.5, 1.0, 1.0])

    antecedents = [{'low': low_membership, 'medium': medium_membership, 'high': high_membership} for _ in range(9)]

    edge = {
        'low': low_membership,
        'high': high_membership,
        'yes': lambda x: cp.full_like(x, cp.float32(0.5))
    }

    rows, cols = fuzzy_image.shape
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1), 
        (0, -1),  (0, 0),  (0, 1),  
        (1, -1),  (1, 0),  (1, 1)   
    ]

    di = cp.array([n[0] for n in neighbors])
    dj = cp.array([n[1] for n in neighbors])

    expanded_image = cp.pad(fuzzy_image, pad_width=1, mode='constant', constant_values=0)
    i_coords, j_coords = cp.meshgrid(cp.arange(1, rows - 1), cp.arange(1, cols - 1), indexing='ij')
    neighbor_values = cp.array([expanded_image[i_coords + di[n], j_coords + dj[n]] for n in range(9)])

    for idx in range(9):
        membership_low = antecedents[idx]['low'](neighbor_values[idx])
        membership_med = antecedents[idx]['medium'](neighbor_values[idx])
        membership_high = antecedents[idx]['high'](neighbor_values[idx])
        
        final_values = cp.where((membership_high > membership_med) & 
                                (membership_high > membership_low), cp.float32(1.0), fuzzy_image[i_coords + di[idx], j_coords + dj[idx]])

        final_values = cp.where((membership_med > membership_high) & 
                                (membership_med > membership_low), cp.float32(0.5), final_values)

        final_values = cp.where((membership_low > membership_med) & 
                                (membership_low > membership_high), cp.float32(0.0), final_values)

        fuzzy_image[i_coords + di[idx], j_coords + dj[idx]] = final_values

    expanded_image = cp.pad(fuzzy_image, pad_width=1, mode='constant', constant_values=0)
    neighbor_values = cp.array([expanded_image[i_coords + di[n], j_coords + dj[n]] for n in range(9)])

    universo = cp.linspace(0, 1, 256)
    rules_output = define_rules(antecedents, edge, neighbor_values, universo)
    
    edge_image = cp.zeros((rows, cols), dtype=cp.float32)
    edge_image[i_coords, j_coords] = cp.where(rules_output > 0, rules_output, 0)

    edge_image_uint8 = cp.clip(edge_image * 255, 0, 255).astype(cp.uint8)
    white_contours = cp.where(edge_image_uint8 > 0, 255, 0).astype(cp.uint8)
    
    return white_contours
