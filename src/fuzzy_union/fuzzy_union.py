import numpy as cp


def triangular(x, a, b, c):
    """
    Función de pertenencia triangular o trapezoidal (hombro/shoulder).
    
    Garantiza que todas las salidas pertenezcan estrictamente al intervalo [0.0, 1.0],
    incluso para valores que superen el rango superior (ej. peso > 300).
    
    Parámetros:
    x : float o array-like
        Valor o valores de entrada.
    a : float
        Inicio de la función.
    b : float
        Punto de máxima pertenencia (pico).
    c : float
        Fin de la función.
        
    Retorna:
    float o cp.ndarray
        Valor de pertenencia en [0.0, 1.0].
    """
    is_scalar = not isinstance(x, cp.ndarray)
    x_arr = cp.asarray(x, dtype=cp.float32)
    
    if a == b:
        # Hombro izquierdo: 1.0 para x <= b, desciende linealmente a 0 en c, 0.0 para x >= c
        res = cp.clip((c - x_arr) / float(c - b), 0.0, 1.0)
    elif b == c:
        # Hombro derecho: 0.0 para x <= a, asciende linealmente a 1 en b, 1.0 para x >= b
        res = cp.clip((x_arr - a) / float(b - a), 0.0, 1.0)
    else:
        # Función triangular estándar
        term1 = (x_arr - a) / float(b - a)
        term2 = (c - x_arr) / float(c - b)
        res = cp.clip(cp.minimum(term1, term2), 0.0, 1.0)
        
    return float(res) if is_scalar else res


def fuzzy_and(a, b):
    """Operador AND difuso (mínimo t-norma)."""
    return cp.min((a, b))


def fuzzy_or(a, b):
    """Operador OR difuso (máximo s-norma)."""
    return cp.max((a, b))


def apply_rules(distancia, y_centroid, weight):
    """
    Evalúa el sistema de reglas difusas para clasificar un candidato como obstáculo.
    
    Las reglas utilizan factores de ponderación (multiplicadores de regla):
    - Reglas 1-5 (multiplicador * 3): Coincidencia espacial cercana y tamaño significativo.
    - Reglas 6-8 (multiplicador * 2): Coincidencia espacial o tamaño relevante.
    - Regla 9 (multiplicador * 1): Tamaño grande por sí solo.
    
    Retorna:
    float: Puntuación ponderada (weighted score / puntaje de activación ponderado) en el rango [0, 3].
    Nota: Este valor representa la fuerza de activación acumulada ponderada de las reglas,
    no una función de pertenencia normalizada entre cero y uno.
    """
    rule1 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["bottom"]), weight["big"]) * 3
    rule2 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["below"]), weight["big"]) * 3
    rule3 = fuzzy_and(fuzzy_and(distancia["average"], y_centroid["below"]), weight["big"]) * 3
    rule4 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["bottom"]), weight["medium"]) * 3
    rule5 = fuzzy_and(fuzzy_and(distancia["near"], y_centroid["below"]), weight["medium"]) * 3
    rule6 = fuzzy_and(distancia["near"], weight["big"]) * 2
    rule7 = fuzzy_and(distancia["average"], weight["big"]) * 2
    rule8 = fuzzy_and(distancia["near"], weight["medium"]) * 2
    rule9 = weight["big"]

    score = fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(fuzzy_or(rule1, rule2), rule3), rule4), rule5), rule6), rule7), rule8), rule9)
    return float(score)


def fuzzy_union(cuadros_list, lidar=(0.0, 0.0), crop_offsets=None):
    """
    Función que recibe una lista de listas de regiones candidatas de cada detector
    (ej. [cuadros_rgb, cuadros_cmeans]) y devuelve la unión difusa con las mejores regiones.
    
    Parámetros:
    cuadros_list: list of lists
        Lista donde cada elemento es una lista de regiones:
        [
            [{"x_centroid": float, "y_centroid": float, "weight": int, ...}, ...], # Detector 1 (ej. RGB)
            [{"x_centroid": float, "y_centroid": float, "weight": int, ...}, ...]  # Detector 2 (ej. FCM)
        ]
        IMPORTANTE: Las coordenadas x_centroid e y_centroid deben estar referenciadas
        a la imagen completa, habiendo compensado previamente el recorte del horizonte.
    lidar: tuple (angulo, distancia)
        Datos opcionales del sensor LiDAR.
    crop_offsets: list of int, opcional
        Desfases verticales [offset_0, offset_1] a sumar a y_init, y_end, y_centroid
        si aún no fueron compensados previamente.

    Retorna:
    list: Lista con el mejor cuadro seleccionado por cada detector (o None si no hay candidatos),
    cada uno con su atributo 'fuzzy_union' que contiene la puntuación ponderada [0, 3].
    """
    # Hacer copia profunda superficial para no mutar destructivamente las listas externas
    cuadros_list_proc = [[dict(c) for c in detector_cuadros] for detector_cuadros in cuadros_list]

    # Compensar desfases verticales si fueron especificados y no estaban compensados
    if crop_offsets is not None:
        for idx, offset in enumerate(crop_offsets):
            if idx < len(cuadros_list_proc) and offset > 0:
                for cuadro in cuadros_list_proc[idx]:
                    if not cuadro.get("_compensated", False):
                        cuadro["y_init"] = cuadro.get("y_init", 0) + offset
                        cuadro["y_end"] = cuadro.get("y_end", 0) + offset
                        cuadro["y_centroid"] = cuadro.get("y_centroid", 0) + offset
                        cuadro["_compensated"] = True

    num_detectores = len(cuadros_list_proc)

    # 1. Calcular distancia mínima hacia los candidatos de los otros detectores
    # usando los centroides espaciales reales (x_centroid, y_centroid)
    for i in range(num_detectores):
        cuadros_i = cuadros_list_proc[i]
        
        # Reunir todos los candidatos de los demás detectores
        otros_candidatos = []
        for j in range(num_detectores):
            if i != j:
                otros_candidatos.extend(cuadros_list_proc[j])
                
        for cuadro in cuadros_i:
            if len(otros_candidatos) == 0:
                # Si no hay candidatos del otro detector, se define una distancia finita
                # de 100.0 (cota superior de saturación del conjunto difuso 'far'), evitando
                # pasar 'inf' a las funciones de pertenencia.
                cuadro["distancia_minima"] = 100.0
            else:
                dist_min = 100.0
                xc1 = cuadro["x_centroid"]
                yc1 = cuadro["y_centroid"]
                
                for otro in otros_candidatos:
                    xc2 = otro["x_centroid"]
                    yc2 = otro["y_centroid"]
                    
                    if lidar != (0.0, 0.0):
                        ang1 = cp.atan2(yc1, xc1)
                        ang2 = cp.atan2(yc2, xc2)
                        if abs(lidar[0] - ang1) < 0.1 or abs(lidar[0] - ang2) < 0.1:
                            dist = 0.0
                        else:
                            dist = float(cp.sqrt((xc1 - xc2)**2 + (yc1 - yc2)**2))
                    else:
                        dist = float(cp.sqrt((xc1 - xc2)**2 + (yc1 - yc2)**2))
                        
                    if dist < dist_min:
                        dist_min = dist
                        
                cuadro["distancia_minima"] = min(dist_min, 100.0)

    # 2. Fuzzificación y aplicación de reglas difusas ponderadas
    resultado = []
    for i in range(num_detectores):
        max_cuadro = None
        for k, cuadro in enumerate(cuadros_list_proc[i]):
            near = triangular(cuadro["distancia_minima"], 0, 0, 20)
            average = triangular(cuadro["distancia_minima"], 10, 25, 40)
            far = triangular(cuadro["distancia_minima"], 30, 100, 100)

            half = triangular(cuadro["y_centroid"], 0, 0, 30)
            below = triangular(cuadro["y_centroid"], 20, 40, 60)
            bottom = triangular(cuadro["y_centroid"], 50, 250, 250)

            small = triangular(cuadro["weight"], 0, 0, 50)
            medium = triangular(cuadro["weight"], 40, 60, 80)
            big = triangular(cuadro["weight"], 70, 300, 300)

            score_val = apply_rules(
                {"far": far, "average": average, "near": near},
                {"half": half, "below": below, "bottom": bottom},
                {"big": big, "medium": medium, "small": small}
            )
            cuadro["fuzzy_union"] = score_val
            if k < len(cuadros_list[i]):
                cuadros_list[i][k]["fuzzy_union"] = score_val
                cuadros_list[i][k]["distancia_minima"] = cuadro.get("distancia_minima", 100.0)

            if max_cuadro is None or cuadro["fuzzy_union"] > max_cuadro["fuzzy_union"]:
                max_cuadro = cuadro

        resultado.append(max_cuadro)

    return resultado


def boxes_intersect(b1, b2, tol=0):
    """
    Verifica si dos bounding boxes se intersectan espacialmente en el plano 2D,
    con una tolerancia opcional (tol >= 0) en píxeles.
    """
    if b1 is None or b2 is None:
        return False
    return not (
        b1["x_end"] + tol < b2["x_init"]
        or b2["x_end"] + tol < b1["x_init"]
        or b1["y_end"] + tol < b2["y_init"]
        or b2["y_end"] + tol < b1["y_init"]
    )


def compute_box_intersection(b1, b2):
    """
    Calcula el rectángulo geométrico de intersección entre dos bounding boxes.
    Retorna None si no hay intersección física.
    """
    if b1 is None or b2 is None:
        return None
    xi = max(b1["x_init"], b2["x_init"])
    yi = max(b1["y_init"], b2["y_init"])
    xe = min(b1["x_end"], b2["x_end"])
    ye = min(b1["y_end"], b2["y_end"])
    if xe > xi and ye > yi:
        return {
            "x_init": int(xi),
            "y_init": int(yi),
            "x_end": int(xe),
            "y_end": int(ye),
            "width": int(xe - xi),
            "height": int(ye - yi),
            "weight": int((xe - xi) * (ye - yi)),
            "x_centroid": float((xi + xe) / 2.0),
            "y_centroid": float((yi + ye) / 2.0)
        }
    return None


def intersect_fuzzy_detections(cuadros_fcm, cuadros_union, tol=0, min_score=0.0):
    """
    Confirma obstáculos cuadro a cuadro mediante la INTERSECCIÓN espacial entre
    las regiones candidatas del FCM y las detecciones de la Unión Difusa / detector complementario,
    eliminando la necesidad de persistencia o memoria temporal inter-frame.

    Parámetros:
    cuadros_fcm: list of dict
        Candidatos generados por el detector semántico FCM.
    cuadros_union: list of dict
        Candidatos generados por el detector complementario / fusión difusa.
    tol: int
        Tolerancia espacial en píxeles (0 = solapamiento físico 2D estricto).
    min_score: float
        Puntuación difusa mínima requerida para el candidato de unión.

    Retorna:
    list of dict:
        Lista de obstáculos confirmados por consenso e intersección.
    """
    confirmed = []
    valid_fcm = [b for b in cuadros_fcm if b is not None]
    valid_union = [
        b for b in cuadros_union
        if b is not None and b.get("fuzzy_union", 0.0) >= min_score
    ]

    for bf in valid_fcm:
        for bu in valid_union:
            if boxes_intersect(bf, bu, tol=tol):
                inter_geom = compute_box_intersection(bf, bu)
                score = max(bf.get("fuzzy_union", 0.0), bu.get("fuzzy_union", 0.0))

                c_conf = dict(bf)
                c_conf["fuzzy_union"] = float(score)
                c_conf["intersected_with"] = {
                    "x_init": bu["x_init"],
                    "y_init": bu["y_init"],
                    "x_end": bu["x_end"],
                    "y_end": bu["y_end"],
                    "score": bu.get("fuzzy_union", 0.0)
                }
                if inter_geom is not None:
                    c_conf["intersection_box"] = inter_geom

                confirmed.append(c_conf)
                break

    return confirmed


if __name__ == "__main__":
    # Comprobaciones mínimas solicitadas:
    # 1. Dos regiones con el mismo centroide dan distancia cero.
    reg_a = [{'x_centroid': 100, 'y_centroid': 80, 'weight': 150, 'x': 15000, 'y': 12000}]
    reg_b = [{'x_centroid': 100, 'y_centroid': 80, 'weight': 600, 'x': 60000, 'y': 48000}]
    res = fuzzy_union([reg_a, reg_b])
    print("Test 1 - Mismo centroide:")
    print("  Distancia reg_a:", reg_a[0].get("distancia_minima", res[0]["distancia_minima"]))
    print("  Distancia reg_b:", reg_b[0].get("distancia_minima", res[1]["distancia_minima"]))
    assert res[0]["distancia_minima"] == 0.0, "La distancia debe ser 0.0 para centroides iguales"

    # 2. Un peso de 600 no produce pertenencia negativa
    m_big = triangular(600, 70, 300, 300)
    print("Test 2 - Peso de 600 en conjunto 'big':", m_big)
    assert 0.0 <= m_big <= 1.0, f"Pertenencia fuera de rango [0, 1]: {m_big}"
    assert m_big == 1.0, f"Se esperaba 1.0 para peso saturado, se obtuvo {m_big}"

    # 3. Sin candidatos del otro detector: distancia no es infinita
    reg_solo = [{'x_centroid': 120, 'y_centroid': 90, 'weight': 80}]
    res_solo = fuzzy_union([reg_solo, []])
    print("Test 3 - Sin candidatos del otro detector:")
    print("  Distancia:", res_solo[0]["distancia_minima"])
    print("  Puntuación ponderada:", res_solo[0]["fuzzy_union"])
    assert res_solo[0]["distancia_minima"] == 100.0, "La distancia debe ser 100.0 (finita) sin candidatos"

    # 4. Comprobación de intersección de cajas
    b_fcm_test = {"x_init": 100, "y_init": 50, "x_end": 150, "y_end": 90, "weight": 200, "fuzzy_union": 1.5}
    b_rgb_test = {"x_init": 120, "y_init": 60, "x_end": 160, "y_end": 95, "weight": 180, "fuzzy_union": 1.2}
    assert boxes_intersect(b_fcm_test, b_rgb_test), "Deben intersectar"
    conf = intersect_fuzzy_detections([b_fcm_test], [b_rgb_test])
    assert len(conf) == 1, "Debe confirmar 1 obstáculo por intersección"
    print("Test 4 - Intersección exitosa:", conf[0]["intersection_box"])

    print("\nTodos los tests mínimos pasaron exitosamente.")