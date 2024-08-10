import cupy as cp
from collections import OrderedDict
from .term import Term  # Asegúrate de que el archivo term.py esté en el mismo directorio o en el path adecuado
from .ordered_dict import OrderedDict

class FuzzyVariable(object):
    """
    Base class containing universe variable & associated membership functions.
    """

    def __init__(self, universe, label, defuzzify_method='centroid'):
        """
        Initialization of fuzzy variable.

        Parameters
        ----------
        universe : array-like
            Universe variable. Must be 1-dimensional and convertible to a CuPy
            array.
        label : string
            Unique name of the universe variable, e.g., 'food' or 'velocity'.
        """
        self.universe = universe
        self.label = label
        self.defuzzify_method = defuzzify_method
        self.terms = OrderedDict()

        self._id = id(self)

    def __setitem__(self, key, item):
        """
        Enable terms to be added with the syntax::

          variable['new_label'] = new_mf
        """
        if isinstance(item, Term):
            if item.label != key:
                raise ValueError("Term's label must match new key")
            if item.parent is not None:
                raise ValueError("Term must not already have a parent")
        else:
            # Try to create a term from item, assuming it is a membership
            # function
            item = Term(key, item)

        mf = item.mf

        if mf.size != self.universe.size:
            raise ValueError("New membership function {0} must be equivalent "
                             "in length to the universe variable.\n"
                             "Expected {1}, got {2}.".format(
                                 key, self.universe.size, mf.size))

        if (mf.max() > 1. + 1e-6) or (mf.min() < 0 - 1e-6):
            raise ValueError("Membership function {0} contains values out of "
                             "range. Allowed range is [0, 1].".format(key))

        # If above pass, add the new membership function
        item.parent = self
        self.terms[key] = item