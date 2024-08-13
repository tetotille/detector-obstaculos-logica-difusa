import cupy as cp
import matplotlib.pyplot as plt

# Funciones de Membresía
def triangular(x, a, b, c):
    x = cp.asarray(x)
    if b == a:
        left = cp.zeros_like(x)
    else:
        left = (x - a) / (b - a)
        
    if c == b:
        right = cp.zeros_like(x)
    else:
        right = (c - x) / (c - b)
    
    return cp.maximum(cp.minimum(left, right), 0)

# Fuzzificación
def fuzzify(value, membership_functions):
    memberships = {}
    value = cp.asarray(value)
    for label, func in membership_functions.items():
        memberships[label] = func(value)
    return memberships

# Inferencia Difusa
def apply_rules(memberships, rules, output_functions):
    output_memberships = {}
    for rule in rules:
        antecedent, consequent = rule
        min_membership = cp.min(cp.array([memberships[var] for var in antecedent]))
        if consequent not in output_memberships:
            output_memberships[consequent] = min_membership * output_functions[consequent]
        else:
            output_memberships[consequent] = cp.maximum(output_memberships[consequent], min_membership * output_functions[consequent])
    return output_memberships

# Defuzzificación y Visualización con función triangular
def defuzzify(output_memberships, universo):
    numerator = 0.0
    denominator = 0.0
    for label, membership in output_memberships.items():
        print(f"Membresía '{label}':", membership)  # Imprimir membresías de salida
        numerator += cp.sum(universo * membership)
        denominator += cp.sum(membership)
    
    print("Denominador:", denominator)  # Imprimir denominador
    
    # Visualizar el resultado con una función triangular
    plt.figure()
    for label, membership in output_memberships.items():
        plt.plot(cp.asnumpy(universo), cp.asnumpy(membership), label=f"Output - {label}")
    
    plt.xlabel('Input Variable')
    plt.ylabel('Membership Degree')
    plt.title('Output Membership Function')
    plt.legend()
    plt.show()
    
    return numerator / denominator if denominator != 0 else 0

# Ejemplo de uso completo
universo = cp.linspace(0, 10, 100)

# Definir funciones de membresía de entrada
membership_functions = {
    'low': lambda x: triangular(x, 0, 0, 5),
    'medium': lambda x: triangular(x, 3, 5, 7),
    'high': lambda x: triangular(x, 5, 10, 10)
}

# Definir funciones de membresía de salida
output_functions = {
    'low': triangular(universo, 0, 2, 4),
    'medium': triangular(universo, 3, 5, 7),
    'high': triangular(universo, 6, 8, 10)
}

# Reglas difusas
rules = [
    (['low'], 'low'),
    (['medium'], 'medium'),
    (['high'], 'high')
]

# Fuzzificación
input_value = 4.7  # Valor flotante
memberships = fuzzify(input_value, membership_functions)

# Aplicación de reglas
output_memberships = apply_rules(memberships, rules, output_functions)

# Defuzzificación y visualización
result = defuzzify(output_memberships, universo)
print("Resultado defuzzificado:", result)
