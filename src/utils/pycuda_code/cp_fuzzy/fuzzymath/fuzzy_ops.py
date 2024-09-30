import cupy as cp

def interp_membership(x, xmf, xx, zero_outside_x=True):
    """
    Find the degree of membership ``u(xx)`` for a given value of ``x = xx`` using CuPy.

    Parameters
    ----------
    x : 1d array
        Independent discrete variable vector.
    xmf : 1d array
        Fuzzy membership function for ``x``.  Same length as ``x``.
    xx : float or array of floats
        Value(s) on universe ``x`` where the interpolated membership is
        desired.
    zero_outside_x : bool, optional
        Defines the behavior if ``xx`` contains value(s) which are outside the
        universe range as defined by ``x``.  If `True` (default), all
        extrapolated values will be zero.  If `False`, the first or last value
        in ``x`` will be what is returned to the left or right of the range,
        respectively.

    Returns
    -------
    xxmf : float or array of floats
        Membership function value at ``xx``, ``u(xx)``.  If ``xx`` is a single
        value, this will be a single value; if it is an array or iterable the
        result will be returned as a CuPy array of like shape.

    Notes
    -----
    For use in Fuzzy Logic, where an interpolated discrete membership function
    u(x) for discrete values of x on the universe of ``x`` is given. Then,
    consider a new value x = xx, which does not correspond to any discrete
    values of ``x``. This function computes the membership value ``u(xx)``
    corresponding to the value ``xx`` using linear interpolation.

    """
    # Not much beats CuPy's built-in interpolation
    if not zero_outside_x:
        left = None
        right = None
    else:
        left = 0.0
        right = 0.0

    return cp.interp(xx, x, xmf, left=left, right=right)

# Ejemplo de uso
x = cp.array([0, 1, 2, 3, 4])
xmf = cp.array([0, 0.2, 0.5, 0.7, 1.0])
xx = cp.array([1.5, 2.5, 3.5])

result = interp_membership(x, xmf, xx)
print(result)
