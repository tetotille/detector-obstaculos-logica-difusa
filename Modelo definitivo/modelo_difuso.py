import cupy as cp
from collections import defaultdict

# Función para calcular la distancia euclidiana entre los centros de los cuadrados
def calcular_distancia(cuadro1, cuadro2):
    x1, y1, l1 = cuadro1  # Cuadro1: (x, y, block_size)
    x2, y2, l2 = cuadro2  # Cuadro2: (x, y, block_size)

    # Calcular los centros de los dos cuadrados
    centro1 = (x1 + l1 / 2, y1 + l1 / 2)
    centro2 = (x2 + l2 / 2, y2 + l2 / 2)

    # Calcular la distancia euclidiana entre los centros
    distancia = cp.sqrt((centro1[0] - centro2[0])**2 + (centro1[1] - centro2[1])**2)
    return distancia

# Función para agrupar los cuadrados cercanos en base a una distancia umbral
def agrupar_cuadrados(cuadrados, umbral_distancia):
    grupos = defaultdict(list)  # Diccionario para almacenar los grupos de cuadrados
    visitados = set()  # Set para llevar registro de los cuadrados ya procesados

    # Convertir los cuadros de cupy.ndarray a tuplas hashables
    cuadrados_hashables = [
        (int(cuadro[0]), int(cuadro[1]), int(cuadro[2]))  # Convertir a tupla (x, y, block_size)
        for cuadro in cuadrados
    ]

    # Función auxiliar para hacer agrupamiento mediante DFS (búsqueda en profundidad)
    def agrupar_recursivo(cuadro_actual, grupo_actual):
        grupo_actual.append(cuadro_actual)
        visitados.add(tuple(cuadro_actual))  # Agregar a visitados como tupla

        # Filtrar cuadrados que tienen las mismas coordenadas
        x_actual, y_actual, _ = cuadro_actual
        cuadrados_coincidentes = [
            cuadro for cuadro in cuadrados_hashables
            if (cuadro[0] == x_actual and cuadro[1] == y_actual) and cuadro not in visitados
        ]

        # Verificar distancia solo entre cuadrados coincidentes
        for cuadro in cuadrados_coincidentes:
            if calcular_distancia(cuadro_actual, cuadro) <= umbral_distancia:
                agrupar_recursivo(cuadro, grupo_actual)

    grupo_id = 0
    # Iterar sobre todos los cuadrados
    for cuadro in cuadrados_hashables:
        if cuadro not in visitados:
            grupo_actual = []
            agrupar_recursivo(cuadro, grupo_actual)
            grupos[grupo_id] = grupo_actual
            grupo_id += 1

    return grupos

# Función para verificar si hay coincidencias en un grupo
def tiene_coincidencias(grupo):
    coordenadas = [(cuadro[0], cuadro[1]) for cuadro in grupo]  # (x, y)
    coincidencias = defaultdict(int)

    # Contar coincidencias de cada coordenada
    for coord in coordenadas:
        coincidencias[coord] += 1

    # Comprobar si hay al menos una coincidencia
    return any(count > 1 for count in coincidencias.values())

# Función principal para detectar el objeto principal
def detectar_objeto_principal(cuadrados, umbral_distancia=50):
    # Agrupar los cuadrados que estén lo suficientemente cerca
    grupos = agrupar_cuadrados(cuadrados, umbral_distancia)
    
    # Validar grupos
    grupos_validos = []
    
    for grupo in grupos.values():
        if len(grupo) > 1 and tiene_coincidencias(grupo):
            grupos_validos.append(grupo)

    # Condiciones para el caso de hasta 8 cuadrados...
