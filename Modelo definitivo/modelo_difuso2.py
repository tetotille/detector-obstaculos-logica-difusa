import cupy as cp

def agrupar_cuadrados(lista_cuadrados, max_repeticiones=4):
    # Asegúrate de que lista_cuadrados es un array 2D
    if isinstance(lista_cuadrados, list):
        # Convierte a un array de cupy y garantiza que todos los elementos sean enteros
        lista_cuadrados_array = cp.array([[int(val) if isinstance(val, cp.ndarray) else val for val in cuadrado]
                                          for cuadrado in lista_cuadrados])
    else:
        raise ValueError("lista_cuadrados debe ser una lista de listas o una matriz 2D.")
    
    # Verificar que tenga la forma esperada
    if lista_cuadrados_array.ndim != 2 or lista_cuadrados_array.shape[1] < 4:
        raise ValueError("lista_cuadrados debe tener al menos 4 columnas.")

    # Paso 1: Obtener las coordenadas de los cuadrados como un array
    coords = lista_cuadrados_array[:, :4].astype(cp.int32)

    # Paso 2: Contar repeticiones usando un hashable (tupla)
    coords_hash = cp.unique(coords, axis=0, return_counts=True)
    max_repeticiones_encontradas = cp.max(coords_hash[1])
    if max_repeticiones_encontradas > max_repeticiones:
        # Si la cantidad máxima de repeticiones es mayor que el límite
        mask_max_repeats = coords_hash[1] >= max_repeticiones
        coordenadas_repetidas = coords_hash[0][mask_max_repeats]
    else:
        coordenadas_repetidas = coords_hash[0]

    # Paso 3: Vectorizar la verificación de cercanía
    distancias = cp.sum(cp.abs(coords[:, cp.newaxis, :] - coordenadas_repetidas[cp.newaxis, :, :]), axis=2)
    
    # Definir un umbral de cercanía
    umbral_cercania = 20

    # Encontrar los índices de los cuadrados que están cerca
    cerca_mask = distancias < umbral_cercania

    # Paso 4: Agrupar los cuadrados cercanos
    grupos = []
    for i in range(coordenadas_repetidas.shape[0]):
        if cp.any(cerca_mask[:, i]):  # Verifica si hay alguna distancia menor al umbral
            grupos.append(coordenadas_repetidas[i])

    # Convertir a un array de coordenadas únicas
    coordenadas_unicas = cp.unique(cp.array(grupos), axis=0)

    return coordenadas_unicas


