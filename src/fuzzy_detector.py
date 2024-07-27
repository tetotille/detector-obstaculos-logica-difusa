from abc import ABC, abstractmethod

class FuzzyDetector(ABC):
    def __init__(self):
        self.antecedent = None
        self.consequent = None
        self.rules = []
        self.control_system = None
        self.control_system_simulation = None
        self.kernel = None

    @abstractmethod
    def define_membership_functions(self):
        raise NotImplementedError

    @abstractmethod
    def define_rules(self):
        raise NotImplementedError

    @abstractmethod
    def fuzzify(self, input_value):
        raise NotImplementedError

    @abstractmethod
    def defuzzify(self):
        raise NotImplementedError

    @abstractmethod
    def show_model(self):
        raise NotImplementedError