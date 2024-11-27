import numpy as cp


def triangular(x, a, b, c):
    """
    Triangular membership function.
    
    Parameters:
    x : array-like
        Input values.
    a : float
        Start of the triangle.
    b : float
        Peak of the triangle.
    c : float
        End of the triangle.
    
    Returns:
    array-like
        Membership values.
    """
    if a == b:
        return cp.where(x <= c, 1 - cp.abs(x - b) / (c - b), 0)
    if b == c:
        return cp.where(x >= a, 1 - cp.abs(x - b) / (b - a), 0)
    return cp.maximum(0, cp.minimum((x - a) / (b - a), (c - x) / (c - b)))

def fuzzy_and(a, b):
    """
    Fuzzy AND operator.
    
    Parameters:
    a : array-like
        First input.
    b : array-like
        Second input.
    
    Returns:
    array-like
        Fuzzy AND values.
    """
    return cp.min((a, b))

def fuzzy_or(a, b):
    """
    Fuzzy OR operator.
    
    Parameters:
    a : array-like
        First input.
    b : array-like
        Second input.
    
    Returns:
    array-like
        Fuzzy OR values.
    """
    return cp.max((a, b))

def apply_rules(distancia, y_centroid, weight):
    
    # Reglas
    # 1. Si la distancia es cerca, el y_centroid es bottom y el peso es grande, entonces es un obstáculo
    # 2. Si la distancia es cerca, el y_centroid es below y el peso es grande, entonces es un obstáculo
    # 3. Si la distancia es media, el y_centroid es below y el peso es grande, entonces es un obstáculo
    # 4. Si la distancia es cerca, el y_centroid es bottom y el peso es mediano, entonces es un obstáculo
    # 5. Si la distancia es cerca, el y_centroid es below y el peso es mediano, entonces es un obstáculo
    # 6. Si la distancia es cerca y el peso es grande, entonces es un obstáculo
    # 7. Si la distancia es media y el peso es grande, entonces es un obstáculo
    # 8. Si la distancia es cerca y el peso es medio, entonces es un obstáculo
    # 9. Si el peso es grande, entonces es un obstáculo

    rule1 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["bottom"]), weight["big"])*3
    rule2 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["below"]), weight["big"])*3
    rule3 = fuzzy_and(fuzzy_and(distancia["average"], y_centroid["below"]), weight["big"])*3
    rule4 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["bottom"]), weight["medium"])*3
    rule5 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["below"]), weight["medium"])*3
    rule6 = fuzzy_and(distancia["near"], weight["big"])*2
    rule7 = fuzzy_and(distancia["average"], weight["big"])*2
    rule8 = fuzzy_and(distancia["near"], weight["medium"])*2
    rule9 = weight["big"]

    return fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(rule1, rule2), rule3), rule4), rule5), rule6), rule7), rule8), rule9)

def fuzzy_union(cuadros_list,lidar=(0.0,0.0)):
    """
    Función que recibe una lista de cuadros y devuelve la unión difusa de los cuadros

    cuadros: [[
        {
            "x": int,
            "y": int,
            weight: int
        },...
    ],...]
    lidar: [float,float]
    """
    for _ in range(len(cuadros_list)):
        cuadros = cuadros_list.pop(0)
        for cuadro in cuadros:
            if "distancia_minima" not in cuadro:
                cuadro["distancia_minima"] = float('inf')
            for cuadros2 in cuadros_list:
                for cuadro2 in cuadros2:
                    distancia = cp.sqrt((cuadro["x"] - cuadro2["x"])**2 + (cuadro["y"] - cuadro2["y"])**2)
                    cuadro["distancia_minima"] = min(cuadro["distancia_minima"], distancia,100)
        cuadros_list.append(cuadros)

    for cuadros in cuadros_list:
        for cuadro in cuadros:
            near = triangular(cuadro["distancia_minima"], 0, 0, 20)
            average = triangular(cuadro["distancia_minima"], 10, 25, 40)
            far = triangular(cuadro["distancia_minima"], 30, 100, 100)

            half = triangular(cuadro["y_centroid"], 0, 0, 30)
            below = triangular(cuadro["y_centroid"], 20, 40, 60)
            bottom = triangular(cuadro["y_centroid"], 50, 250, 250)

            small = triangular(cuadro["weight"], 0, 0, 50)
            medium = triangular(cuadro["weight"], 40, 60, 80)
            big = triangular(cuadro["weight"], 70, 300, 300)

            cuadro["fuzzy_union"] = apply_rules({"far":far,
                                                 "average":average,
                                                 "near":near,},
                                                 {"half":half,
                                                  "below":below,
                                                  "bottom":bottom,},
                                                  {"big":big,
                                                   "medium":medium,
                                                   "small":small,})
    return cuadros_list

if __name__ == "__main__":
    cuadros_rgb = [{'puntos': None, 'x_init': 71, 'x_end': 81, 'y_init': 8, 'y_end': 18, 'x': 2882, 'y': 497, 'weight': 38, 'x_centroid': 75, 'y_centroid': 13}, {'puntos': None, 'x_init': 175, 'x_end': 215, 'y_init': 45, 'y_end': 51, 'x': 19489, 'y': 4760, 'weight': 100, 'x_centroid': 194, 'y_centroid': 47}, {'puntos': None, 'x_init': 229, 'x_end': 249, 'y_init': 52, 'y_end': 54, 'x': 7224, 'y': 1585, 'weight': 30, 'x_centroid': 240, 'y_centroid': 52}]
    cuadros_cmeans = [{'puntos': None, 'x_init': 4, 'x_end': 242, 'y_init': 0, 'y_end': 22, 'x': 445261, 'y': 39261, 'weight': 4219, 'x_centroid': 105, 'y_centroid': 9}, {'puntos': None, 'x_init': 93, 'x_end': 109, 'y_init': 57, 'y_end': 68, 'x': 10097, 'y': 6203, 'weight': 100, 'x_centroid': 100, 'y_centroid': 62}]

    fuzzy_union_image = fuzzy_union([cuadros_rgb, cuadros_cmeans])
    print(fuzzy_union_image)