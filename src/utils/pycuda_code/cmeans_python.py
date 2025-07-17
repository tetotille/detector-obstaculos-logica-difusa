"""
cmeans.py : Fuzzy C-means clustering algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist
from os.path import dirname, abspath, join
from sys import argv
import cv2

def _distance(data, centers, metric='euclidean'):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers, metric=metric).T

def normalize_columns(columns):
    """
    Normalize columns of matrix.

    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns

    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns/np.sum(columns, axis=0, keepdims=1)
    
    return normalized_columns


def normalize_power_columns(x, exponent):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.

    Parameters
    ----------
    x : 2d array (M x N)
        Matrix with columns
    n : float
        Exponent

    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**n) but safe
    
    """

    assert np.all(x >= 0.0)

    x = x.astype(np.float64)
    
    # values in range [0, 1]
    x = x/np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)
        
        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x**exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x**exponent

    result = normalize_columns(x)

    return result


def _cmeans0(data, u_old, c, m, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)
    print(u_old)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    print(data)
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T
    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d

if __name__ == "__main__":
    # Parámetros de prueba
    num_clusters = 3
    m = 2.0

    # Cargar y procesar la imagen
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
    img = cv2.imread(filename)
    img = cv2.resize(img, (200, int(img.shape[0] * 200 / img.shape[1])))
    pixels = img.astype(np.float32) / 255.0
    
    # Reshape la imagen para que cada píxel sea una fila y los valores RGB sean las columnas
    data = np.reshape(pixels, (-1, 3))
    num_data = data.shape[0]
    
    # Generar u_old con la forma correcta
    u_old = np.random.rand(num_data, num_clusters).astype(np.float32)

    # Ejecuta la función _cmeans0
    cntr,u, jm_value, d = _cmeans0(data, u_old, num_clusters, m, metric='euclidean')

    # Imprime resultados
    print("Cluster Centers (cntr):")
    print(cntr)
    print("\nDistances (d):")
    print(d)
    print("\nObjective Function (jm):")
    print(jm_value)

