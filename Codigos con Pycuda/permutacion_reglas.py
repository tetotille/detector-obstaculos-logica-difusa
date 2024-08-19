import cupy as cp

def triangular(x, abc):
    a, b, c = abc
    return cp.maximum(0, cp.minimum((x - a) / (b - a), (c - x) / (c - b)))

def define_membership_functions(image):
    min_pixel = cp.min(image)
    max_pixel = cp.max(image)
    universo = cp.linspace(0, 1, 256)

    # Crear funciones de membresía
    low_membership = lambda x: triangular(x, [0, 0, 0.5])
    medium_membership = lambda x: triangular(x, [max_pixel / 3, (min_pixel + max_pixel) / 2, max_pixel * 2 / 3])
    high_membership = lambda x: triangular(x, [0.5, 1.0, 1.0])

    antecedents = [{'low': low_membership, 'medium': medium_membership, 'high': high_membership} for _ in range(9)]

    # Definición de edge con todas las funciones de membresía relevantes
    edge = {
        'low': low_membership,
        'medium': medium_membership,
        'high': high_membership,
        'yes': lambda x: cp.float32(0.5) if isinstance(x, (int, float)) else cp.full_like(x, cp.float32(0.5))  # Definición de 'yes' como un singleton
    }

    return antecedents, edge


def permutar_reglas(reglas):
    """
    Recibe una lista de reglas y devuelve una lista con las reglas permutadas.
    Cada regla es una tupla (left, middle, right, edge_value).
    """
    reglas_permutadas = []

    for regla in reglas:
        left, middle, right, edge_value = regla
        
        # Primera permutación: izquierda -> medio, medio -> derecha, derecha -> izquierda
        permutacion1 = (middle, right, left, edge_value)
        
        # Segunda permutación: izquierda -> derecha, medio -> izquierda, derecha -> medio
        #permutacion2 = (right, left, middle, edge_value)
        
        # Tercera permutación: derecha -> medio, medio -> izquierda, izquierda -> derecha
        #permutacion3 = (left, middle, right, edge_value)
        
        # Añadir las permutaciones a la lista
        reglas_permutadas.append(permutacion1)
        #reglas_permutadas.append(permutacion2)
        #reglas_permutadas.append(permutacion3)
    
    return reglas_permutadas

def formatear_regla(regla):
    left, middle, right, edge_value = regla
    # Convertir la salida edge_value a un string correspondiente
    edge_str = 'yes' if edge_value(0) == 0.5 else 'low' if edge_value(0) == 0 else 'high'
    return f"({left}, {middle}, {right}, {edge_str})"

def mostrar_permutaciones(reglas_permutadas):
    for i, regla in enumerate(reglas_permutadas):
        regla_formateada = formatear_regla(regla)
        print(f"Permutación {i + 1}: {regla_formateada}")

# Ejemplo de uso

# Supongamos que tienes una imagen definida (por ejemplo, una imagen de 256x256)
image = cp.random.rand(256, 256)

# Define las funciones de membresía y edge
antecedents, edge = define_membership_functions(image)

# Define las reglas que quieres permutar
reglas = [
    ([0, 1, 3, 4, 6, 7], [], [2, 5, 8], edge['yes']),
    ([0, 1, 2, 3, 4, 5], [], [6, 7, 8], edge['yes']),
    ([3, 4, 5, 6, 7, 8], [], [0, 1, 2], edge['yes']),
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
    # Agrega aquí las demás reglas según sea necesario
]

# Realiza las permutaciones
reglas_permutadas = permutar_reglas(reglas)

# Muestra las permutaciones formateadas
mostrar_permutaciones(reglas_permutadas)
