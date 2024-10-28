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
        # Convertir las coordenadas en una tupla hashable accediendo a los valores numéricos
        coord = (cuadrado[0].item(), cuadrado[1].item(), cuadrado[2], cuadrado[3])
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

    # Paso 2: Crear un grupo inicial con el cuadrado más repetido
    grupo_principal = cuadrado_mas_repetido if cuadrado_mas_repetido else []

    # Paso 3: Agregar cuadrados cercanos al grupo principal de manera iterativa
    while True:
        nuevos_agregados = False
        
        # Verificamos cada cuadrado en el grupo actual
        for cuadrado in grupo_principal[:]:  # Hacemos una copia para evitar modificar mientras iteramos
            for cuadrado_otro in lista_cuadrados:
                # Asegurarnos de que cuadrado_otro no esté ya en grupo_principal
                if cuadrado_otro not in grupo_principal:
                    if son_cercanos(cuadrado_otro, cuadrado):
                        grupo_principal.append(cuadrado_otro)
                        nuevos_agregados = True  # Marcar que hemos hecho una adición

        # Si no se han agregado nuevos cuadrados, terminamos
        if not nuevos_agregados:
            break

    # Crear un conjunto para almacenar coordenadas únicas
    coordenadas_unicas = set()
    for cuadrado in grupo_principal:
        coord = (cuadrado[0].item(), cuadrado[1].item())
        coordenadas_unicas.add(coord)

    # Enviar coordenadas únicas al final
    return list(coordenadas_unicas)  # Convertir de nuevo a lista si es necesario

# Ejemplo de uso
lista_cuadrados = [
    # Aquí deberías incluir tus coordenadas
]

resultado = agrupar_cuadrados(lista_cuadrados)
print("Coordenadas únicas agrupadas:", resultado)
