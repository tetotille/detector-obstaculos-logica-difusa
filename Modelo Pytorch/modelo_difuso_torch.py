import torch

def agrupar_cuadrados(lista_cuadrados, max_repeticiones=2):
    # Asegúrate de que lista_cuadrados es un array 2D
    if isinstance(lista_cuadrados, list):
        # Convierte a un tensor de PyTorch y garantiza que todos los elementos sean enteros
        lista_cuadrados_tensor = torch.tensor([[val.item() if isinstance(val, torch.Tensor) else val for val in cuadrado]
                                                for cuadrado in lista_cuadrados], dtype=torch.int32)
    else:
        raise ValueError("lista_cuadrados debe ser una lista de listas o un tensor 2D.")
    
    # Verificar que tenga la forma esperada
    if lista_cuadrados_tensor.ndim != 2 or lista_cuadrados_tensor.shape[1] < 4:
        raise ValueError("lista_cuadrados debe tener al menos 4 columnas.")

    # Paso 1: Obtener las coordenadas de los cuadrados como un tensor
    coords = lista_cuadrados_tensor[:, :4]

    # Paso 2: Contar repeticiones usando un hashable (tupla)
    coords_hash, counts = torch.unique(coords, dim=0, return_counts=True)

    # Encontrar el índice del cuadrado más repetido
    max_repeticiones_encontradas = counts.max()
    
    if max_repeticiones_encontradas >= max_repeticiones:  # Al menos 'max_repeticiones' veces
        indice_max = counts.argmax()
        coordenada_mas_repetida = coords_hash[indice_max]
        print(f"Coordenada más repetida: {coordenada_mas_repetida.tolist()}, repeticiones: {max_repeticiones_encontradas.item()}")

        # Paso 3: Agrupar los cuadrados que son adyacentes a la coordenada más repetida
        umbral_cercania = 20  # Definir un umbral de cercanía
        distancias = torch.sum(torch.abs(coords - coordenada_mas_repetida), dim=1)

        # Encontrar los índices de los cuadrados que están cerca de la coordenada más repetida
        cerca_mask = distancias < umbral_cercania

        # Agrupar los cuadrados cercanos
        grupos = coords[cerca_mask]

        # Convertir a un tensor de coordenadas únicas
        coordenadas_unicas = torch.unique(grupos, dim=0)
        print("Coordenadas únicas agrupadas:", coordenadas_unicas.tolist())

        # Devolver las coordenadas únicas como listas de tensores de PyTorch
        resultado = [coord for coord in coordenadas_unicas]
        return resultado
    else:
        print("No se encontraron suficientes repeticiones para agrupar.")
        return []  # Retorna una lista vacía si no hay suficientes repeticiones
