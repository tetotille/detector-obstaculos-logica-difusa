def replace_words(rules, old_words, new_words):
    """
    Reemplaza palabras en las reglas difusas.

    Parameters:
    rules (list of str): Lista de reglas en formato de cadena.
    old_words (list of str): Lista de palabras a reemplazar.
    new_words (list of str): Lista de palabras nuevas.

    Returns:
    list of str: Lista de reglas con las palabras reemplazadas.
    """
    replaced_rules = []
    for rule in rules:
        for old, new in zip(old_words, new_words):
            rule = rule.replace(old, new)
        replaced_rules.append(rule)
    return replaced_rules

# Lista de reglas originales
rules=[
    """rule1 = ctrl.Rule(C1['high'] & C3['high'] & C5['high'] & 
                      C2['high'] & C4['high'] & C6['high'] & 
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule2 = ctrl.Rule(C5['high'] & C7['high'] & C9['high'] & 
                      C1['low'] & C2['low'] & C3['low'] & 
                      C4['high'] & C6['high'] & C8['high'], edge['yes'])
    
    rule3 = ctrl.Rule(C5['high'] & C4['low'] & C1['low'] & 
                      C9['high'] & C2['high'] & C7['low'] & 
                      C3['high'] & C6['high'] & C8['high'], edge['yes'])
    
    rule4 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] & 
                      C9['low'] & C6['low'] & C3['low'] & 
                      C5['high'] & C7['high'] & C8['high'], edge['yes'])
    
    rule5 = ctrl.Rule(C1['low'] & C2['low'] & C4['low'] & 
                      C5['low'] & C6['high'] & C3['high'] & 
                      C7['high'] & C9['high'] & C8['high'], edge['yes'])
    
    rule6 = ctrl.Rule(C2['high'] & C1['high'] & C3['high'] & 
                      C8['low'] & C6['high'] & C9['high'] & 
                      C4['low'] & C5['low'] & C7['low'], edge['yes'])
    
    rule7 = ctrl.Rule(C4['high'] & C7['high'] & C8['high'] & 
                     C1['high'] & C6['low'] & C3['low'] & 
                     C2['low'] & C9['high'] & C5['low'], edge['yes'])
    
    rule8 = ctrl.Rule(C6['low'] & C5['low'] & C8['low'] & 
                     C1['high'] & C4['high'] & C3['high'] & 
                     C2['high'] & C9['low'] & C7['high'], edge['yes'])
    
    rule9 = ctrl.Rule(C4['low'] & C5['low'] & C1['low'] & 
                     C6['high'] & C8['high'] & C9['high'] & 
                     C2['low'] & C3['low'] & C7['low'], edge['yes'])
    
    rule10 = ctrl.Rule(C2['low'] & C5['low'] & C9['low'] & 
                     C1['low'] & C6['low'] & C3['low'] & 
                     C4['high'] & C8['high'] & C7['high'], edge['yes'])
    
    rule11 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] & 
                     C5['low'] & C6['low'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])
    
    # ... añadir las otras reglas aquí

    rule12 = ctrl.Rule(C1['low'] & C5['low'] & C9['low'] & 
                       C2['high'] & C3['high'] & C4['low'] & 
                       C6['high'] & C7['low'] & C8['low'], edge['yes'])
    
    rule13 =ctrl.Rule(C1['low'] & C5['low'] & C9['low'] & 
                       C2['low'] & C3['low'] & C4['low'] & 
                       C6['low'] & C7['low'] & C8['low'], edge['low'])
    
    rule14 = ctrl.Rule(C5['high'] & C3['high'] & C1['high'] & 
                      C9['high'] & C2['high'] & C7['high'] & 
                      C4['high'] & C6['high'] & C8['high'], edge['low'])
    
    rule15 = ctrl.Rule(C5['low'] & C3['low'] & C1['low'] & 
                      C2['low'] & C4['low'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule16= ctrl.Rule(C5['low'] & C3['low'] & C6['low'] & 
                      C9['low'] & C8['low'] & C2['low'] & 
                      C1['high'] & C4['high'] & C7['high'], edge['yes']) 
    
    rule17= ctrl.Rule(C5['low'] & C4['low'] & C6['low'] & 
                      C9['low'] & C8['low'] & C7['low'] & 
                      C1['high'] & C2['high'] & C3['high'], edge['yes']) 
    
    rule18 = ctrl.Rule(C9['high'] & C3['high'] & C6['high'] & 
                      C7['low'] & C8['low'] & C5['low'] & 
                      C1['low'] & C2['low'] & C3['low'], edge['low'])  
    
    #otro tipo de division
  
    rule19 = ctrl.Rule(C1['high'] & C2['high'] & C4['high'] & 
                      C5['high'] & C6['low'] & C3['low'] & 
                      C7['low'] & C9['low'] & C8['low'], edge['yes'])
    
    rule20 = ctrl.Rule(C2['low'] & C1['low'] & C3['low'] & 
                      C8['high'] & C6['low'] & C9['low'] & 
                      C4['high'] & C5['high'] & C7['high'], edge['yes']) 
    
    rule21 = ctrl.Rule(C4['low'] & C7['low'] & C8['low'] & 
                     C1['low'] & C6['high'] & C3['high'] & 
                     C2['high'] & C9['low'] & C5['high'], edge['yes'])
    
    rule22 = ctrl.Rule(C6['high'] & C5['high'] & C8['high'] & 
                     C1['low'] & C4['low'] & C3['low'] & 
                     C2['low'] & C9['high'] & C7['low'], edge['yes'])
    #Aca otro
    rule23 = ctrl.Rule(C4['high'] & C5['high'] & C1['high'] &   #cambiar
                     C6['low'] & C8['low'] & C9['low'] & 
                     C2['high'] & C3['low'] & C7['low'], edge['yes'])
    
    rule24 = ctrl.Rule(C5['high'] & C6['high'] & C1['high'] &  
                     C4['low'] & C8['low'] & C7['low'] & 
                     C2['high'] & C3['high'] & C9['high'], edge['yes'])
    
    rule25 = ctrl.Rule(C3['high'] & C5['high'] & C9['high'] & 
                     C1['low'] & C2['low'] & C4['low'] & 
                     C6['high'] & C8['high'] & C7['high'], edge['yes'])
    
    rule26 = ctrl.Rule(C1['high'] & C5['high'] & C4['high'] & 
                     C2['high'] & C6['low'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['yes'])
    
    rule27 = ctrl.Rule(C1['low'] & C5['high'] & C4['high'] & 
                     C2['low'] & C6['high'] & C3['low'] & 
                     C8['low'] & C9['low'] & C7['low'], edge['low'])
    
    rule28 = ctrl.Rule(C1['low'] & C5['low'] & C4['high'] &     #cambiar
                     C2['low'] & C6['high'] & C3['low'] & 
                     C8['high'] & C9['high'] & C7['high'], edge['yes'])
    
    rule29= ctrl.Rule(C1['low'] & C5['high'] & C4['low'] & 
                     C2['high'] & C6['low'] & C3['low'] & 
                     C8['high'] & C9['low'] & C7['low'], edge['low'])   
    
    rule30= ctrl.Rule(C1['high'] & C5['low'] & C4['high'] & 
                     C2['low'] & C6['high'] & C3['high'] & 
                     C8['low'] & C9['high'] & C7['high'], edge['low'])   
    
    rule31 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] & 
                      C4['high'] & C5['low'] & C6['high'] & 
                      C7['low'] & C8['low'] & C9['low'], edge['yes'])
    

    rule32 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] & 
                      C4['high'] & C5['low'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['low'], edge['yes'])
    # Definir la regla para cuando solo una variable es 'low' y el resto son 'high'
    rule33 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] & 
                      C4['low'] & C5['low'] & C6['high'] & 
                      C7['low'] & C8['high'] & C9['high'], edge['yes'])

# Definir la regla para cuando solo una variable es 'high' y el resto son 'low'
    rule34 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] & 
                      C4['high'] & C5['high'] & C6['low'] & 
                      C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule35 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])

    rule36 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])

    rule37 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])

    rule38 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule39 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    
    rule40 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule41 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule42 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    
    rule43 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['low'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    
    rule44 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['low'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    
    rule45 = ctrl.Rule(C1['low'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['high'], edge['yes'])
    
    rule46 = ctrl.Rule(C1['low'] & C2['low'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['high'] & C9['high'], edge['yes'])
    
    rule47 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule48 = ctrl.Rule(C1['low'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['high'], edge['yes'])
    
    rule49 = ctrl.Rule(C1['high'] & C2['low'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['high'] & C9['low'], edge['yes'])
    
    rule50 = ctrl.Rule(C1['high'] & C2['high'] & C3['low'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['high'] & C8['low'] & C9['low'], edge['yes'])
    
    rule51 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['low'] & C5['high'] & C6['high'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])
    
    rule52 = ctrl.Rule(C1['high'] & C2['high'] & C3['high'] &
                   C4['high'] & C5['high'] & C6['low'] &
                   C7['low'] & C8['low'] & C9['low'], edge['yes'])"""]

# Palabras a reemplazar
old_words = ['high']
new_words = ['medium']

# Reemplazar las palabras en las reglas
new_rules = replace_words(rules, old_words, new_words)

# Imprimir las nuevas reglas
for rule in new_rules:
    print(rule)
