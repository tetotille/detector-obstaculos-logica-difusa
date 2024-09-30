def define_rules(antecedents, edge, neighbor_values, universo, output_file="reglas_activadas.txt"):
    num_rules = len(antecedents)
    num_neighbors = neighbor_values.shape[0]

    # Definir las reglas en formato de listas de índices para high, medium, low y la salida
    rule_sets = [
        ([0, 1, 3, 4, 6, 7], [], [2, 5, 8], edge['yes']),  # Ejemplo de regla 1
        ([0, 1, 2, 3, 4, 5], [], [6, 7, 8], edge['yes']),  # Ejemplo de regla 2
        ([3, 4, 5, 6, 7, 8], [], [0, 1, 2], edge['yes']),  # Ejemplo de regla 3
                ([1, 2, 4, 5, 7, 8], [], [0, 4, 6], edge['yes']), #4
        ([2, 5, 6, 7, 8], [], [0, 1, 3, 4], edge['yes']), #5
        ([0, 1, 2, 5, 8], [], [3, 4, 6, 7], edge['yes']), #6
        ([0, 3, 6, 7, 8], [], [1, 2, 4, 5], edge['yes']), #7
        ([0, 1, 2, 3, 6], [], [4, 5, 7, 8], edge['yes']), #8
        ([5, 7, 8], [], [0, 1, 2, 3, 4, 6], edge['yes']), #9
        ([3, 6, 7], [], [0, 1, 2, 4, 5, 8], edge['yes']), #10
        ([0, 1, 3], [], [2, 4, 5, 6, 7, 8], edge['yes']), #11
        ([1, 2, 5], [], [0, 3, 4, 6, 7, 8], edge['yes']), #12
    #create_rule([], [], [0, 3, 4, 6, 1, 2, 5, 7, 8], edge['low']) #13
    #create_rule([0, 3, 4, 6, 1, 2, 5, 7, 8], [], [], edge['low']) #14
        ([6, 7, 8], [], [0, 1, 2, 3, 4, 5], edge['yes']), #15
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']), #16
        ([0, 1, 2], [], [3, 4, 5, 6, 7, 8], edge['yes']), #17
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']), #18
        ([2, 5, 8], [], [0, 1, 3, 4, 6, 7], edge['yes']), #19
        ([3, 4, 6, 7], [], [0, 1, 2, 5, 8], edge['yes']), #20
        ([0, 3, 6], [], [1, 2, 4, 5, 7, 8], edge['yes']), #21
        ([4, 5, 7, 8], [], [0, 1, 2, 3, 6], edge['yes']), #22
        ([0, 1, 3, 4], [], [2, 5, 6, 7, 8], edge['yes']), #23
        ([0, 1, 2, 4, 5, 8], [], [3, 6, 7], edge['yes']), #24
        ([2, 4, 5, 6, 7, 8], [], [0, 1, 3], edge['yes']), #25
        ([0, 1, 2, 3, 4, 6], [], [5, 7, 8], edge['yes']), #26
        ([3, 4, 5], [], [0, 1, 2, 6, 7, 8], edge['yes']), #27
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']), #28
    #create_rule([1, 4, 7], [], [0, 2, 3, 5, 6, 8], edge['yes']) #29
    #create_rule([0, 2, 3, 5, 6, 8], [], [1, 4, 7], edge['yes']) #30
        ([0, 1, 2, 3, 5], [], [4, 6, 7, 8], edge['yes']), #31
        ([0, 1, 3, 6, 7], [], [2, 4, 5, 8], edge['yes']), #32
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']), #33
        ([0, 3, 4, 6, 7, 8], [], [1, 2, 5], edge['yes']), #34
        ([0, 3, 4, 5], [], [1, 2, 5, 7, 8], edge['yes']), #35
        ([4, 6, 7, 8], [], [0, 1, 2, 3, 5], edge['yes']), #36
        ([2, 4, 5, 8], [], [0, 1, 3, 6, 7], edge['yes']), #37
        ([0, 1, 2, 4], [], [3, 5, 6, 7, 8], edge['yes']), #38
        ([0, 3, 6, 7], [], [1, 2, 4, 5, 8], edge['yes']), #39
        ([3, 6, 7, 8], [], [0, 1, 2, 4, 5], edge['yes']), #40
        ([5, 6, 7, 8], [], [0, 1, 2, 3, 4], edge['yes']), #41
        ([0, 1, 3, 4, 6], [], [2, 5, 7, 8], edge['yes']), #42
        ([3, 5, 6, 7, 8], [], [0, 1, 2, 4], edge['yes']), #43
        ([0, 1, 3, 6], [], [2, 4, 5, 7, 8], edge['yes']), #44
        ([1, 2, 4, 5, 8], [], [0, 3, 6, 7], edge['yes']), #45
        ([2, 4, 5, 7, 8], [], [0, 1, 3, 6], edge['yes']), #46
        ([4, 5, 6, 7, 8], [], [0, 1, 2, 3], edge['yes']), #47
        ([3, 4, 6, 7, 8], [], [0, 1, 2, 5], edge['yes']), #48
        ([0, 3, 4, 6, 7], [], [1, 2, 5, 8], edge['yes']), #49
        ([0, 1, 3, 4, 6], [], [2, 5, 7, 8], edge['yes']), #50
        ([0, 1, 2, 4, 5], [], [3, 6, 7, 8], edge['yes']), #51
        ([0, 1, 2, 3, 4], [], [5, 6, 7, 8], edge['yes']), #52
        # Agrega más reglas según sea necesario...
    ]

    # Convertir a matriz de reglas
    indices_high = [rule[0] for rule in rule_sets]
    indices_medium = [rule[1] for rule in rule_sets]
    indices_low = [rule[2] for rule in rule_sets]
    outputs = [rule[3] for rule in rule_sets]

    # Expandir neighbor_values para que cada conjunto corresponda a una regla
    expanded_neighbor_values = cp.repeat(neighbor_values[:, cp.newaxis, :, :], len(rule_sets), axis=1)

    # Calcular memberships para cada conjunto de reglas
    high_memberships = cp.ones_like(expanded_neighbor_values[0])
    medium_memberships = cp.ones_like(expanded_neighbor_values[0])
    low_memberships = cp.ones_like(expanded_neighbor_values[0])

    for i, indices in enumerate(indices_high):
        for idx in indices:
            high_memberships[i] = cp.minimum(high_memberships[i], antecedents[idx]['high'](expanded_neighbor_values[idx, i]))

    for i, indices in enumerate(indices_medium):
        for idx in indices:
            medium_memberships[i] = cp.minimum(medium_memberships[i], antecedents[idx]['medium'](expanded_neighbor_values[idx, i]))

    for i, indices in enumerate(indices_low):
        for idx in indices:
            low_memberships[i] = cp.minimum(low_memberships[i], antecedents[idx]['low'](expanded_neighbor_values[idx, i]))

    # Calcular la membresía mínima entre high, medium y low para cada regla
    final_memberships = cp.minimum(cp.minimum(high_memberships, medium_memberships), low_memberships)

    # Evaluar cuáles reglas se activan y guardar los resultados para la defuzzificación
    rules = []
    for i in range(len(rule_sets)):
        activation = final_memberships[i] > 0.4

        if cp.any(activation):
            with open(output_file, "a") as f:
                f.write(f"Regla {i + 1} activada con final_membership:\n")
                np_final_membership = final_memberships[i].get()  # Convertir a NumPy para imprimir
                np.set_printoptions(precision=10, suppress=False, floatmode='fixed')  # Ajustar precisión decimal
                f.write(np.array2string(np_final_membership, separator=', ') + "\n")
            
            # Guardar la membresía final y el valor de salida correspondiente
            rules.append((final_memberships[i], outputs[i]))

    # Procesar las reglas activadas y realizar la defuzzificación
    final_crisp_values = defuzzify_centroid(rules, universo)
    
    print("Final Crisp Values:")
    print(final_crisp_values.get())  # Utiliza .get() para obtener el array en formato NumPy desde CuPy
    
    max_value = cp.max(final_crisp_values).get()  # Obtener el valor máximo utilizando CuPy y convertirlo a NumPy
    print("Máximo valor del array final_crisp_values:", max_value)

    with open(output_file, "a") as f:
        f.write(f"Salida defuzzificada por píxel: {final_crisp_values.get()}\n")

    # Retornar la salida defuzzificada
    return final_crisp_values