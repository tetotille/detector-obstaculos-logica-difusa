import cupy as cp
import cv2

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
def defuzzify_centroid(min_value, output_function, universo):
    if cp.any(min_value > 0):  # Verificar si hay algún valor mayor que 0 en min_value
        numerator = cp.sum(min_value * output_function(universo))
        denominator = cp.sum(min_value)
        if denominator != 0:
            return numerator / denominator
        else:
            return cp.zeros_like(universo[0])  # Retorna 0 si el denominador es 0
    else:
        return cp.zeros_like(universo[0])  # Retorna 0 si no hay valores significativos en min_value



# Definición de funciones de membresía
def define_membership_functions(image):
    min_pixel = cp.min(image)
    max_pixel = cp.max(image)
    universo = cp.linspace(0, 1, 256)

    # Crear funciones de membresía
    low_membership = lambda x: triangular(x, [0, 0, 0.5])
    medium_membership = lambda x: triangular(x, [max_pixel / 3, (min_pixel + max_pixel) / 2, max_pixel * 2 / 3])
    high_membership = lambda x: triangular(x, [0.5, 1.0, 1.0])

    antecedents = [{'low': low_membership, 'medium': medium_membership, 'high': high_membership} for _ in range(9)]

    edge = {
        'low': low_membership,
        'high': high_membership,
        'yes': lambda x: triangular(x, [0.5, 0.5, 0.5])  # Definición de 'yes' como un singleton
    }

    return antecedents, edge, universo

# Definición de reglas difusas
def define_rules(antecedents, edge, neighbor_values):
    rules = []

    def create_rule(indices_high, indices_medium, indices_low, output):
        memberships = []

        if indices_high:
            high_memberships = [antecedents[i]['high'](neighbor_values[i]) for i in indices_high]
            memberships.extend(high_memberships)
        
        if indices_medium:
            medium_memberships = [antecedents[i]['medium'](neighbor_values[i]) for i in indices_medium]
            memberships.extend(medium_memberships)
        
        if indices_low:
            low_memberships = [antecedents[i]['low'](neighbor_values[i]) for i in indices_low]
            memberships.extend(low_memberships)

        if memberships:
            min_membership = memberships[0]
            for m in memberships[1:]:
                min_membership = cp.minimum(min_membership, m)
            rules.append((min_membership, output))

    create_rule([0], [], [], edge['yes'])  # Regla que lleva a 'yes'
    create_rule([0], [3], [4], edge['low'])  # Regla que lleva a 'low'

    return rules

# Aplicar reglas difusas a la imagen

# Aplicar reglas difusas a la imagen
def apply_fuzzy_rules_to_image(fuzzy_image, rules, antecedents):
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
            membership_values['low'] = cp.zeros_like(neighbor_values[idx])
            membership_values['medium'] = cp.zeros_like(neighbor_values[idx])
            membership_values['high'] = cp.zeros_like(neighbor_values[idx])
            membership_values_list.append(membership_values)
    
    # Aplicar actualizaciones a la imagen difusa en el vecino central (índice 4)
    central_idx = 4
    membership_values = membership_values_list[central_idx]
    max_values = cp.stack([membership_values['low'], membership_values['medium'], membership_values['high']], axis=0)
    max_labels = cp.argmax(max_values, axis=0)

    # Verificar si se debe actualizar la imagen difusa solo en el píxel central
    high_values = cp.where(max_labels == 2, cp.float32(1.0), fuzzy_image[i_coords + di[central_idx], j_coords + dj[central_idx]])
    medium_values = cp.where(max_labels == 1, cp.float32(0.5), fuzzy_image[i_coords + di[central_idx], j_coords + dj[central_idx]])

    fuzzy_image[i_coords + di[central_idx], j_coords + dj[central_idx]] = cp.maximum(high_values, medium_values)

    write_fuzzy_image_to_file(fuzzy_image, filename="fuzzy_image_values.txt")
    print("Forma de edge_image:", fuzzy_image.shape)

    # Aplicar reglas al vecino central únicamente (índice 4)
    central_pixel_output = cp.zeros_like(fuzzy_image[i_coords, j_coords])

    # Recorrer todas las reglas y determinar el valor de salida defuzzificado para el píxel central
    for rule_idx, (rule, output) in enumerate(rules):
        min_membership = rule
        min_value = cp.min(min_membership, axis=0)

        # Aquí se aplica la defuzzificación si el output es una función de membresía
        if callable(output):
            output_value = defuzzify_centroid(min_value, output, cp.linspace(0, 1, 256))
        else:
            output_value = output

        # Asignar el valor defuzzificado al píxel central si es mayor que el valor actual
        central_pixel_output = cp.maximum(central_pixel_output, output_value)

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
    print(f"Valores mínimos y máximos de la imagen en escala de grises: min={min_val}, max={max_val}")

    if max_val > min_val:  # Para evitar división por cero
        grayscale_image_np = (grayscale_image - min_val) / (max_val - min_val) * 255.0
    else:
        print("Advertencia: min_val es igual a max_val, no se puede normalizar.")
        
    grayscale_image_numpy = cp.clip(grayscale_image_np, 0, 255).astype(cp.uint8)

    # Convertir la imagen de CuPy a NumPy para visualizarla con OpenCV
    grayscale_image_numpy = grayscale_image_numpy.get()

    # Visualizar la imagen en escala de grises usando OpenCV
    cv2.imshow('Imagen en escala de grises', grayscale_image_numpy)
    cv2.waitKey(0)

    # Continuar con el proceso en CuPy
    x, y = grayscale_image.shape
    resized_image = resize_image(grayscale_image, (int(x * 200 / y), 200))

    # Verificar valores mínimos y máximos después de redimensionar
    min_val_resized = cp.min(resized_image)
    max_val_resized = cp.max(resized_image)
    print(f"Valores mínimos y máximos de la imagen redimensionada: min={min_val_resized}, max={max_val_resized}")
    if max_val_resized > min_val_resized:
        resized_image_np = (resized_image - min_val_resized) / (max_val_resized - min_val_resized) * 255.0
    resized_image_numpy = cp.clip(resized_image_np, 0, 255).astype(cp.uint8)
    
    # Convertir la imagen redimensionada a NumPy para visualizarla
    resized_image_numpy = resized_image_numpy.get()
    
    # Mostrar la imagen redimensionada
    cv2.imshow('Imagen Redimensionada', resized_image_numpy)
    cv2.waitKey(0)

    fuzzy_image = resized_image.astype(cp.float32) / 256.0  # Normalizar la imagen entre 0 y 1
    print(f"Valores mínimos y máximos de la imagen normalizada: min={cp.min(fuzzy_image)}, max={cp.max(fuzzy_image)}")

    # Definir las funciones de membresía
    antecedents, edge, universo = define_membership_functions(fuzzy_image)

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

    # Generar las coordenadas i y j de toda la imagen
    rows, cols = fuzzy_image.shape
    i_coords, j_coords = cp.meshgrid(cp.arange(1, rows-1), cp.arange(1, cols-1), indexing='ij')

    # Calcular los valores de los vecinos para todos los píxeles en paralelo
    neighbor_values = cp.array([expanded_image[i_coords + di[n], j_coords + dj[n]] for n in range(9)])

    # Definir las reglas utilizando neighbor_values calculados
    rules = define_rules(antecedents, edge, neighbor_values)

    # Aplicar las reglas difusas a la imagen
    edge_image = apply_fuzzy_rules_to_image(fuzzy_image, rules, antecedents)
    print("Forma de edge_image:", edge_image.shape)
    write_image_to_file(edge_image, filename="edge_image_values.txt")
    edge_image_uint8 = cp.clip(edge_image * 255, 0, 255).astype(cp.uint8)

    # Retornar la imagen procesada como un array de CuPy
    return edge_image_uint8

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

