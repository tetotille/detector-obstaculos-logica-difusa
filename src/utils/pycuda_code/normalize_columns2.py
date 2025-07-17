import cupy as cp


def normalize_power_columns(matrix, power):
    """
    Normalize columns of the matrix with the given power.

    Parameters
    ----------
    matrix : 2d cupy array
        Matrix to be normalized.
    power : float
        Power to which each column is raised.

    Returns
    -------
    normalized_matrix : 2d cupy array
        Column-normalized matrix.
    """
    powered_matrix = cp.power(matrix, power)
    column_sums = cp.sum(powered_matrix, axis=0)
    normalized_matrix = powered_matrix / column_sums
    return normalized_matrix

def normalize_columns(u):
    """
    Normalize columns of the given matrix.
    
    Parameters
    ----------
    u : 2d cupy array
        Matrix to be normalized.
    
    Returns
    -------
    normalized_u : 2d cupy array
        Column-normalized matrix.
    """
    column_sums = cp.sum(u, axis=0)
    normalized_u = u / column_sums
    return normalized_u