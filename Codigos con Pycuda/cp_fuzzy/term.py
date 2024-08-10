"""
term.py : Framework to create fuzzy terms.

Most notably, contains the Term and WeightedTerm objects which are used to
identify specific membership functions attached to Antecedents or
Consequents when constructing fuzzy Rules.

Terms have redefined logical operators which enable the simple and elegant
combination of several during Rule creation.
"""
from __future__ import print_function, division

import cupy as cp

from .state import StatefulProperty


class TermPrimitive(object):
    """
    Marker class for type checking when a term or term aggregate is expected.
    """

    def membership_value(self):
        raise NotImplementedError("Implement in concrete class")

    def __and__(self, other):
        if not isinstance(other, TermPrimitive):
            raise ValueError("Can only construct 'AND' from the term "
                             "of a fuzzy variable")

        return TermAggregate(self, other, 'and')

    def __or__(self, other):
        if not isinstance(other, TermPrimitive):
            raise ValueError("Can only construct 'OR' from the term "
                             "of a fuzzy variable")

        return TermAggregate(self, other, 'or')

    def __invert__(self):
        return TermAggregate(self, None, 'not')


class Term(TermPrimitive):
    """
    A Term is a universe and associated specific membership function.

    For example, if one were creating a FuzzyVariable with a simple three-
    point liker scale, three Term would be created named poor, average,
    and good.
    """

    def __init__(self, label, membership_function):
        super(Term, self).__init__()
        self.label = label
        self.parent = None
        self.mf = membership_function

    @property
    def full_label(self):
        """Term with parent.  Ex: velocity['fast']"""
        if self.parent is None:
            raise ValueError("This term must be bound to a parent first")
        return self.parent.label + "[" + self.label + "]"

    def __repr__(self):
        return self.full_label
