import cupy as cp

def obtener_centro(cuadrado):
    """Recibe un cuadrado y calcula su centro en coordenadas."""
    x1, y1, width, height = cuadrado
    centro_x = x1 + width / 2
    centro_y = y1 + height / 2
    return cp.array([centro_x, centro_y])

def distancia_centros(c1, c2):
    """Calcula la distancia euclidiana entre dos centros de cuadrados."""
    return cp.linalg.norm(c1 - c2)

def agrupar_cuadrados(cuadrados, umbral_distancia):
    """Agrupa cuadrados que estén cercanos entre sí basándose en la distancia de sus centros."""
    centros = [obtener_centro(cuadrado) for cuadrado in cuadrados]
    grupos = []
    
    while centros:
        grupo_actual = [cuadrados.pop(0)]  # Inicia con el primer cuadrado disponible
        centro_actual = centros.pop(0)
        
        i = 0
        while i < len(centros):
            if distancia_centros(centro_actual, centros[i]) <= umbral_distancia:
                grupo_actual.append(cuadrados.pop(i))
                centros.pop(i)  # Eliminar el centro también para mantener el índice correcto
            else:
                i += 1
        
        grupos.append(grupo_actual)
    
    return grupos

def verificar_y_devolver_grupo_mayor(cuadrados, umbral_distancia):
    """Verifica la coincidencia de cuadrados y agrupa, retornando el grupo más grande."""
    if not cuadrados:
        return None
    
    grupos = agrupar_cuadrados(cuadrados, umbral_distancia)
    
    # Si no hay grupos, no retornar nada
    if not grupos:
        return None
    
    # Ordenar los grupos por tamaño
    grupos_ordenados = sorted(grupos, key=len, reverse=True)
    
    # Comparar tamaños de los dos grupos más grandes
    if len(grupos_ordenados) > 1 and len(grupos_ordenados[0]) == len(grupos_ordenados[1]):
        return None  # Si hay un empate, no retornar nada
    else:
        grupo_mayor = grupos_ordenados[0]  # Retornar el grupo mayor
        
        # Devolver las coordenadas originales de los cuadrados coincidentes (sin duplicados)
        coordenadas_unicas = set()
        for cuadrado in grupo_mayor:
            x1, y1, _, _ = cuadrado  # Obtener las coordenadas originales
            coordenadas_unicas.add((x1.item(), y1.item()))  # Convertir a tupla de valores hashables
        
        return list(coordenadas_unicas)  # Retornar la lista de coordenadas únicas


# Ejemplo de uso:
"""cuadrados = [
    (cp.array(120), cp.array(15), 15, 15),
    (cp.array(120), cp.array(15), 15, 15),
    (cp.array(150), cp.array(15), 15, 15),
    # Agrega más cuadrados según sea necesario
]

umbral_distancia = 15"""  # Asumimos que si la distancia es igual al tamaño del bloque (15), son "cercanos"
