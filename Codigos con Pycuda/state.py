import cupy as cp

class StatefulProperty(object):
    def __init__(self, initial_condition=None):
        self.default = initial_condition
        self.data = {'current': initial_condition}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return self.data[instance]
        except KeyError:
            val = self.default  # Asumimos que default ya es un arreglo de CuPy
            self.data[instance] = val
            return val

    def __set__(self, instance, value):
        raise AttributeError("Property is read-only. Did you mean to access via a simulation?")

    def clear(self, initial_condition=None):
        self.__init__(self.default)

class MyClass:
    state = StatefulProperty(cp.array([0]))  # Inicia con estado 0 en CuPy
