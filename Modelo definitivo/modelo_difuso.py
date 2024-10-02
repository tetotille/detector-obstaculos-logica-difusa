def son_cercanos(cuadrado1, cuadrado2, distancia=15):
    """Determina si dos cuadrados están a una distancia de un bloque."""
    x1, y1 = cuadrado1[0].item(), cuadrado1[1].item()
    x2, y2 = cuadrado2[0].item(), cuadrado2[1].item()
    return abs(x1 - x2) <= distancia and abs(y1 - y2) <= distancia

def agrupar_cuadrados(lista_cuadrados, max_repeticiones=4):
    # Paso 1: Identificar el cuadrado con más repeticiones de coordenadas
    coordenadas_repetidas = {}
    cuadrado_mas_repetido = None
    max_repeticiones_encontradas = 0

    for cuadrado in lista_cuadrados:
        coord = (cuadrado[0].item(), cuadrado[1].item())
        if coord not in coordenadas_repetidas:
            coordenadas_repetidas[coord] = [cuadrado]
        else:
            coordenadas_repetidas[coord].append(cuadrado)

        # Si encontramos más repeticiones que el máximo, actualizamos
        if len(coordenadas_repetidas[coord]) > max_repeticiones_encontradas:
            max_repeticiones_encontradas = len(coordenadas_repetidas[coord])
            cuadrado_mas_repetido = coordenadas_repetidas[coord]

        # Si alcanzamos el máximo permitido de repeticiones, ya no buscamos más
        if max_repeticiones_encontradas == max_repeticiones:
            break

    # Paso 2: Crear grupos de cuadrados con coordenadas repetidas
    grupos = [cuadrados for cuadrados in coordenadas_repetidas.values() if len(cuadrados) > 1]

    # Paso 3: Agregar cuadrados cercanos pero con coordenadas diferentes
    for grupo in grupos:
        cuadrados_no_agrupados = [cuadrado for cuadrado in lista_cuadrados if cuadrado not in sum(grupos, [])]
        
        for cuadrado in cuadrados_no_agrupados[:]:  # Copia de la lista para evitar modificaciones mientras iteramos
            if any(son_cercanos(cuadrado, otro_cuadrado) for otro_cuadrado in grupo):
                grupo.append(cuadrado)
                cuadrados_no_agrupados.remove(cuadrado)

    # Paso 4: Fusionar grupos que tengan cuadrados cercanos en común
    grupos_fusionados = []

    while grupos:
        grupo_actual = grupos.pop(0)
        fusionado = True

        while fusionado:
            fusionado = False
            for otro_grupo in grupos[:]:
                if any(son_cercanos(cuadrado, otro_cuadrado) for cuadrado in grupo_actual for otro_cuadrado in otro_grupo):
                    grupo_actual.extend(otro_grupo)
                    grupos.remove(otro_grupo)
                    fusionado = True
        
        grupos_fusionados.append(grupo_actual)

    # Paso 5: Retornar el grupo que contiene el cuadrado más repetido
    for grupo in grupos_fusionados:
        if cuadrado_mas_repetido[0] in grupo:
            return grupo

    # Si no se encuentra el grupo, retornar el más grande por defecto
    return max(grupos_fusionados, key=len, default=None)
