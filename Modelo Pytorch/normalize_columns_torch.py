import torch

def normalize_power_columns(matrix, power):
    """
    Normaliza las columnas de la matriz con la potencia dada.

    Parameters
    ----------
    matrix : 2d torch tensor
        Matriz a normalizar.
    power : float
        Potencia a la que se eleva cada columna.

    Returns
    -------
    normalized_matrix : 2d torch tensor
        Matriz normalizada por columnas.
    """
    powered_matrix = torch.pow(matrix, power)  # Eleva cada elemento de la matriz a la potencia especificada
    column_sums = torch.sum(powered_matrix, dim=0)  # Suma las columnas
    normalized_matrix = powered_matrix / column_sums  # Normaliza cada columna
    return normalized_matrix

def normalize_columns(u):
    """
    Normaliza las columnas de la matriz dada.
    
    Parameters
    ----------
    u : 2d torch tensor
        Matriz a normalizar.
    
    Returns
    -------
    normalized_u : 2d torch tensor
        Matriz normalizada por columnas.
    """
    column_sums = torch.sum(u, dim=0)  # Suma las columnas
    normalized_u = u / column_sums  # Normaliza cada columna
    return normalized_u