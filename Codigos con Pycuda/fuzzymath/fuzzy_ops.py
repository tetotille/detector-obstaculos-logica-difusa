import cupy as cp

def fuzzy_op(x, a, y, b, op):
    """Operation of two fuzzy sets using CuPy.
    
    Operate fuzzy set ``a`` with fuzzy set ``b``,
    using +, * or any other binary operator.

    Parameters
    ----------
    x : 1d array, length N
        Universe variable for fuzzy set ``a``.
    a : 1d array, length N
        Fuzzy set for universe ``x``.
    y : 1d array, length M
        Universe variable for fuzzy set ``b``.
    b : 1d array, length M
        Fuzzy set for universe ``y``.
    op: Function, pointwise binary operator on two matrices
        (pointwise version of) +, -, *, /, min, max etc.

    Returns
    -------
    z : 1d array
        Output variable.
    mfz : 1d array
        Fuzzy membership set for variable ``z``.

    Notes
    -----
    Uses Zadeh's Extension Principle as described in Ross, Fuzzy Logic with
    Engineering Applications (2010), pp. 414, Eq. 12.17.

    If these results are unexpected and your membership functions are convex,
    consider trying the ``skfuzzy.dsw_*`` functions for fuzzy mathematics
    using interval arithmetic via the restricted Dong, Shah, and Wong method.

    """
    # a and x, and b and y, are formed into (MxN) matrices. The former has
    # identical rows; the latter identical columns.

    yy, xx = cp.meshgrid(y, x, sparse=True)       # consider broadcasting rules
    bb, aa = cp.meshgrid(b, a, sparse=True)

    # Do the operation
    zz = op(xx, yy).ravel()
    zz_index = cp.argsort(zz)
    zz = cp.sort(zz)

    # Array min() operation
    c = cp.fmin(aa, bb).ravel()
    c = c[zz_index]

    # Initialize loop
    z, mfz = cp.zeros(0), cp.zeros(0)
    idx = 0

    for _ in range(len(c)):
        index = cp.nonzero(zz == zz[idx])[0]
        z = cp.hstack((z, zz[idx]))
        mfz = cp.hstack((mfz, c[index].max()))
        idx = index[-1] + 1
        if idx >= len(zz):
            break

    return z, mfz

def _interp_universe_fast(x, xmf, y):
    """
    Find interpolated universe value(s) for a given fuzzy membership value using CuPy.

    Fast version, with possible duplication.

    Parameters
    ----------
    x : 1d array
        Independent discrete variable vector.
    xmf : 1d array
        Fuzzy membership function for ``x``.  Same length as ``x``.
    y : float
        Specific fuzzy membership value.

    Returns
    -------
    xx : list
        List of discrete singleton values on universe ``x`` whose
        membership function value is y, ``u(xx[i])==y``.
        If there are not points xx[i] such that ``u(xx[i])==y``
        it returns an empty list.

    Notes
    -----
    For use in Fuzzy Logic, where a membership function level ``y`` is given.
    Consider there is some value (or set of values) ``xx`` for which
    ``u(xx) == y`` is true, though ``xx`` may not correspond to any discrete
    values on ``x``. This function computes the value (or values) of ``xx``
    such that ``u(xx) == y`` using linear interpolation.
    """
    # Special case required or zero-level cut does not work with faster method
    if y == 0.:
        idx = cp.where(cp.diff(xmf > y))[0]
    else:
        idx = cp.where(cp.diff(xmf >= y))[0]

    # This method is fast, but duplicates point values where
    # y == peak of a membership function.
    return x[idx] + (y - xmf[idx]) * (x[idx+1] - x[idx]) / (xmf[idx+1] - xmf[idx])

# Ejemplo de uso
x = cp.array([1, 2, 3, 4, 5])
xmf = cp.array([0.0, 0.3, 0.7, 0.5, 0.0])
y = 0.5

result = _interp_universe_fast(x, xmf, y)
print(result)
