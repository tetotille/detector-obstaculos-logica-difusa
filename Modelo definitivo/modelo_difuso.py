import cupy as cp


def son_cercanos(cuadrado1, cuadrado2, distancia=15):
    """Determina si dos cuadrados son cercanos entre sí."""
    x1, y1 = cuadrado1[0].get().item(), cuadrado1[1].get().item()
    x2, y2 = cuadrado2[0].get().item(), cuadrado2[1].get().item()
    return abs(x1 - x2) <= distancia and abs(y1 - y2) <= distancia

def agrupar_cuadrados(lista_cuadrados):
    # Diccionario para agrupar por coordenadas iguales
    grupos = {}
    
    # Paso 1: Agrupar cuadrados con coordenadas iguales
    for cuadrado in lista_cuadrados:
        coord = (cuadrado[0].get().item(), cuadrado[1].get().item())
        if coord not in grupos:
            grupos[coord] = [cuadrado]
        else:
            grupos[coord].append(cuadrado)
    
    # Convertir los grupos en una lista
    lista_grupos = list(grupos.values())

    # Paso 2: Fusionar grupos adyacentes
    grupos_fusionados = []

    while lista_grupos:
        grupo_actual = lista_grupos.pop(0)
        fusionado = True
        
        # Intentar fusionar con otros grupos adyacentes
        while fusionado:
            fusionado = False
            for otro_grupo in lista_grupos[:]:
                if any(son_cercanos(cuadrado, otro_cuadrado) for cuadrado in grupo_actual for otro_cuadrado in otro_grupo):
                    grupo_actual.extend(otro_grupo)
                    lista_grupos.remove(otro_grupo)
                    fusionado = True
        
        grupos_fusionados.append(grupo_actual)
    
    # Paso 3: Incluir cuadrados no coincidentes adyacentes
    grupo_final = []
    
    for grupo in grupos_fusionados:
        grupo_final.append(grupo)
        for cuadrado in lista_cuadrados:
            if cuadrado not in sum(grupo_final, []):
                if any(son_cercanos(cuadrado, otro_cuadrado) for otro_cuadrado in sum(grupo_final, [])):
                    grupo.append(cuadrado)
    
    # Paso 4: Fusionar si hay grupos adyacentes a un cuadrado solitario
    fusionados_totales = []

    while grupo_final:
        grupo = grupo_final.pop(0)
        fusionado = False
        for otro_grupo in grupo_final[:]:
            if any(son_cercanos(cuadrado, otro_cuadrado) for cuadrado in grupo for otro_cuadrado in otro_grupo):
                grupo.extend(otro_grupo)
                grupo_final.remove(otro_grupo)
                fusionado = True
        fusionados_totales.append(grupo)
    
    # Retornar el grupo más grande
    grupo_mayor = max(fusionados_totales, key=len, default=None)
    return grupo_mayor



# Ejemplo de uso:
"""cuadrados = [
    (cp.array(120), cp.array(15), 15, 15),
    (cp.array(120), cp.array(15), 15, 15),
    (cp.array(150), cp.array(15), 15, 15),
    # Agrega más cuadrados según sea necesario
]

umbral_distancia = 15"""  # Asumimos que si la distancia es igual al tamaño del bloque (15), son "cercanos"
