import cupy as cp

class OrderedDict:
    """
    Dictionary that remembers insertion order using CuPy for values.
    """

    def __init__(self):
        self._keys = []
        self._values = {}

    def __setitem__(self, key, value):
        if key not in self._values:
            self._keys.append(key)
        self._values[key] = cp.array(value)

    def __delitem__(self, key):
        if key in self._values:
            self._keys = [k for k in self._keys if k != key]
            del self._values[key]
        else:
            raise KeyError(f"Key '{key}' not found")

    def keys(self):
        return self._keys

    def values(self):
        return [self._values[key] for key in self._keys]

    def items(self):
        return [(key, self._values[key]) for key in self._keys]

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._values

    def __repr__(self):
        return f"OrderedDict({{{', '.join(f'{k}: {v}' for k, v in self.items())}}})"

# Ejemplo de uso de OrderedDict con CuPy
od = OrderedDict()
od['one'] = 1
od['two'] = 2
od['three'] = 3

print(od.keys())    # ['one', 'two', 'three']
print(od.values())  # [array(1), array(2), array(3)]
print(od.items())   # [('one', array(1)), ('two', array(2)), ('three', array(3))]
print(len(od))      # 3
print('two' in od)  # True
print(od)           # OrderedDict({'one': array(1), 'two': array(2), 'three': array(3)})
