import cupy as cp

class FuzzyVariable:
    # Placeholder for the actual implementation of FuzzyVariable
    pass

class StatefulProperty:
    # Placeholder para la implementación real de StatefulProperty
    def __init__(self, value):
        self.value = value

class Antecedent(FuzzyVariable):
    """
    Antecedent (input/sensor) variable for a fuzzy control system.

    Parameters
    ----------
    universe : cp.ndarray
        Universe variable. Must be 1-dimensional and in CuPy array format.
    label : string
        Name of the universe variable.
    """

    input = StatefulProperty(None)

    def __init__(self, universe, label):
        """
        Initialize the Antecedent with universe and label.
        """
        super(Antecedent, self).__init__(universe, label)
        self.__name__ = 'Antecedent'
        self.universe = universe
        self.terms = {}  # This should be initialized as needed

    def add_term(self, term_name, term_value):
        """
        Add a term to the Antecedent.
        
        Parameters
        ----------
        term_name : string
            The name of the term to add.
        term_value : cp.ndarray
            The value of the term, must be in CuPy array format.
        """
        self.terms[term_name] = term_value

class Consequent(FuzzyVariable):
    """
    Consequent (output/control) variable for a fuzzy control system.

    Parameters
    ----------
    universe : cp.ndarray
        Universe variable. Must be 1-dimensional and in CuPy array format.
    label : string
        Name of the universe variable.
    defuzzify_method : string
        Name of method used for defuzzification, defaults to 'centroid'

    Notes
    -----
    The ``label`` string chosen must be unique among Antecedents and
    Consequents in the ``ControlSystem``.
    """

    output = StatefulProperty(None)

    def __init__(self, universe, label, defuzzify_method='centroid'):
        """
        Initialize the Consequent with universe, label, and defuzzify_method.
        """
        super(Consequent, self).__init__(universe, label, defuzzify_method)
        self.__name__ = 'Consequent'

    def add_term(self, term_name, term_value):
        """
        Add a term to the Consequent.
        
        Parameters
        ----------
        term_name : string
            The name of the term to add.
        term_value : cp.ndarray
            The value of the term, must be in CuPy array format.
        """
        self.terms[term_name] = term_value

# Ejemplo de uso para Antecedent
universe = cp.array([0, 1, 2, 3, 4, 5])
label = "Temperature"
antecedent = Antecedent(universe, label)

term_value = cp.array([0, 0.5, 1, 0.5, 0])
antecedent.add_term("Cold", term_value)

print(antecedent.terms)

# Ejemplo de uso para Consequent
universe = cp.array([0, 1, 2, 3, 4, 5])
label = "Speed"
consequent = Consequent(universe, label)

term_value = cp.array([0, 0.5, 1, 0.5, 0])
consequent.add_term("Slow", term_value)

print(consequent.terms)
