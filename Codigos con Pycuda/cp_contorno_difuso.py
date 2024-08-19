import cupy as cp
import cv2
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

    return antecedents, edge

def cp_minimum(arr1, arr2):
    # Utiliza una operación de comparación para calcular el mínimo entre dos arrays
    return cp.where(arr1 < arr2, arr1, arr2)

# Definición de reglas difusas
def define_rules(antecedents, edge, neighbor_values, output_file="reglas_activadas.txt"):
    rules = []
    min_kernel = cp.ElementwiseKernel(
    'float32 x, float32 y', 'float32 z',
    'z = min(x, y)',
    'min_kernel')

    def create_rule(indices_high=[], indices_medium=[], indices_low=[], output=None):
        memberships_high = cp.ones_like(neighbor_values[0])
        memberships_medium = cp.ones_like(neighbor_values[0])
        memberships_low = cp.ones_like(neighbor_values[0])

        # Verificación de índices específicos
        if indices_high:
            for i in indices_high:
                if memberships_high.shape == antecedents[i]['high'](neighbor_values[i]).shape:
                    print("Tipo de dato de memberships_high:", memberships_high.dtype)

                    height, width = neighbor_values.shape[1], neighbor_values.shape[2]

                    # Generar 50 índices aleatorios dentro de los límites de neighbor_values
                    random_indices = np.random.choice(height * width, 50, replace=False)

                        # Convertir los índices planos a coordenadas 2D dentro de los límites de neighbor_values
                    random_coords = np.unravel_index(random_indices, (height, width))
                    antecedents_high_values = []

                    for j in range(50):
                        
                        x, y = random_coords[0][j], random_coords[1][j]
                        high_value = antecedents[0]['high'](neighbor_values[0][x, y]).get()  # Cambia el índice [0] si quieres usar otros vecinos
                        antecedents_high_values.append(high_value)
                    high_membership_values = antecedents[i]['high'](neighbor_values[i])
                    memberships_high = cp.minimum(memberships_high, cp.min(antecedents[0]['high'](neighbor_values[0]), axis=0))

                    print("Valores de antecedentes 'high':", antecedents_high_values)
                else:
                    raise ValueError(f"Dimensiones no coinciden: memberships_high {memberships_high.shape}, antecedents {antecedents[i]['high'](neighbor_values[i]).shape}")
            for j in range(9):
                high_values = antecedents[j]['high'](neighbor_values[j])
                print(f"Vecino {j}: high_membership: {high_values.get()}")

        if indices_medium:
            for i in indices_medium:
                memberships_medium = cp.minimum(memberships_medium, antecedents[i]['medium'](neighbor_values[i]))

        if indices_low:
            for i in indices_low:
                memberships_low = cp.minimum(memberships_low, antecedents[i]['low'](neighbor_values[i]))
        
        for i in range(9):
                
            print(f"Valores de neighbor_values para indices_high: {neighbor_values.get()}")
            print(f"memberships_high: {memberships_high.get()}")
            print(f"memberships_medium: {memberships_medium.get()}")
            print(f"memberships_low: {memberships_low.get()}")

        # Combinación final
        final_membership = cp.minimum(cp.minimum(memberships_high, memberships_medium), memberships_low)

        # Activar regla si la membresía mínima es mayor a 0
        if cp.any(final_membership > 0.4):
            with open(output_file, "a") as f:
                f.write(f"Regla activada con final_membership:\n")
                np_final_membership = final_membership.get()  # Convertir a NumPy para imprimir
                np.set_printoptions(precision=10, suppress=False, floatmode='fixed')   # Ajustar precisión decimal
                f.write(np.array2string(np_final_membership, separator=', ') + "\n")
            rules.append((final_membership, output))
        else:
            with open(output_file, "a") as f:
                f.write("Regla no activada debido a final_membership <= 0\n")

    # Ejemplos de reglas
    create_rule([0], [], [], edge['yes'])  # Vecino 0 debe ser high para activar la regla
    #create_rule([0], [3], [4], edge['low'])  # Vecino 0 es high, 3 es medium, y 4 es low para activar la regla

    return rules


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
    print("Forma de edge_image:", fuzzy_image.shape)
    
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
    rules = define_rules(antecedents, edge, neighbor_values)

    # Evaluar reglas y determinar salida
    central_pixel_output = cp.zeros((rows - 2, cols - 2), dtype=cp.float32)

    for rule, output in rules:
        # Verificar si se cumple la regla con los valores actualizados
        min_value = cp.min(rule, axis=0)
        #print(f"min_value: {min_value.get()}")  # Añadir esta línea para verificar los valores

        if callable(output):
            output_value = defuzzify_centroid(min_value, output, cp.linspace(0, 1, 256))
            #print(f"Defuzzificación aplicada, valor: {output_value}")
        else:
            output_value = output

        # Actualizar el valor de salida basado en la regla que se cumple
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
    
    # Normalizar la imagen a un rango de 0 a 255
    min_val = cp.min(grayscale_image)
    max_val = cp.max(grayscale_image)
    if max_val > min_val:  # Para evitar división por cero
        grayscale_image = (grayscale_image - min_val) / (max_val - min_val) * 255.0
    else:
        raise ValueError("Error: min_val es igual a max_val, no se puede normalizar.")
        
    # Redimensionar la imagen (opcional, dependiendo de la necesidad)
    x, y = grayscale_image.shape
    resized_image = resize_image(grayscale_image, (int(x * 200 / y), 200))

    # Normalizar la imagen entre 0 y 1 para procesamiento difuso
    fuzzy_image = resized_image.astype(cp.float32) / 256.0

    # Definir las funciones de membresía
    antecedents, edge = define_membership_functions(fuzzy_image)
    #print_antecedents(antecedents)

    # Aplicar las reglas difusas a la imagen
    edge_image = apply_fuzzy_rules_to_image(fuzzy_image, antecedents, edge)

    write_image_to_file(edge_image, filename="edge_image_values.txt")

    # Convertir la imagen procesada a formato uint8
    edge_image_uint8 = cp.clip(edge_image * 255, 0, 255).astype(cp.uint8)

    # Retornar la imagen procesada
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

