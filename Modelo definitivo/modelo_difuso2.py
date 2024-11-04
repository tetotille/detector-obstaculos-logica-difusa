import cupy as cp

def agrupar_cuadrados(lista_cuadrados, max_repeticiones=2):
    # Asegúrate de que lista_cuadrados es un array 2D
    if isinstance(lista_cuadrados, list):
        # Convierte a un array de cupy y garantiza que todos los elementos sean enteros
        lista_cuadrados_array = cp.array([[val.item() if isinstance(val, cp.ndarray) else val for val in cuadrado]
                                           for cuadrado in lista_cuadrados])
    else:
        raise ValueError("lista_cuadrados debe ser una lista de listas o una matriz 2D.")
    
    # Verificar que tenga la forma esperada
    if lista_cuadrados_array.ndim != 2 or lista_cuadrados_array.shape[1] < 4:
        raise ValueError("lista_cuadrados debe tener al menos 4 columnas.")

    # Paso 1: Obtener las coordenadas de los cuadrados como un array
    coords = lista_cuadrados_array[:, :4].astype(cp.int32)

    # Paso 2: Contar repeticiones usando un hashable (tupla)
    coords_hash, counts = cp.unique(coords, axis=0, return_counts=True)

    # Encontrar el índice del cuadrado más repetido
    max_repeticiones_encontradas = cp.max(counts)
    
    if max_repeticiones_encontradas >= max_repeticiones:  # Al menos 'max_repeticiones' veces
        indice_max = cp.argmax(counts)
        coordenada_mas_repetida = coords_hash[indice_max]
        print(f"Coordenada más repetida: {coordenada_mas_repetida}, repeticiones: {max_repeticiones_encontradas}")

        # Paso 3: Agrupar los cuadrados que son adyacentes a la coordenada más repetida
        umbral_cercania = 20  # Definir un umbral de cercanía
        distancias = cp.sum(cp.abs(coords - coordenada_mas_repetida), axis=1)

        # Encontrar los índices de los cuadrados que están cerca de la coordenada más repetida
        cerca_mask = distancias < umbral_cercania

        # Agrupar los cuadrados cercanos
        grupos = coords[cerca_mask]

        # Convertir a un array de coordenadas únicas
        coordenadas_unicas = cp.unique(grupos, axis=0)
        print("Coordenadas únicas agrupadas:", coordenadas_unicas)

        # Devolver las coordenadas únicas como listas de cupy arrays
        resultado = [cp.array(coord) for coord in coordenadas_unicas]
        return resultado
    else:
        print("No se encontraron suficientes repeticiones para agrupar.")
        return []  # Retorna una lista vacía si no hay suficientes repeticiones
