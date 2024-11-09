import torch
import time
from normalize_columns_torch import normalize_columns, normalize_power_columns
import utils_torch
import cv2
from os.path import join, dirname, abspath

def calculate_distances(data, centers, metric='euclidean'):
    """
    Calculate distances between data points and cluster centers using PyTorch.

    Parameters
    ----------
    data : torch.Tensor
        Data to be analyzed. Shape is (N, Q) where N is the number of data points and Q is the number of features.
    centers : torch.Tensor
        Cluster centers. Shape is (C, Q) where C is the number of clusters and Q is the number of features.
    metric : str
        Metric to use. Currently only 'euclidean' is supported.

    Returns
    -------
    torch.Tensor
        Distance matrix. Shape is (C, N) where C is the number of clusters and N is the number of data points.
    """
    if metric != 'euclidean':
        raise NotImplementedError(f"Metric '{metric}' not implemented.")
    
    # Mover los datos a la GPU si no están ya en ella
    data = data.cuda() if not data.is_cuda else data
    centers = centers.cuda() if not centers.is_cuda else centers

    # Obtener las dimensiones
    N, Q = data.shape
    C = centers.shape[0]

    # Expandir dimensiones para facilitar la operación de diferencia
    data_expanded = data.unsqueeze(1)  # Shape (N, 1, Q)
    centers_expanded = centers.unsqueeze(0)  # Shape (1, C, Q)
    
    # Calcular las diferencias al cuadrado
    diff = data_expanded - centers_expanded  # Shape (N, C, Q)
    sq_dists = torch.sum(diff ** 2, dim=2)  # Shape (N, C)
    
    # Retornar la transpuesta para que tenga la forma esperada (C, N)
    return sq_dists.T


def _cmeans0(data, u_old, c, m, metric='euclidean'):
    # Mover datos a GPU
    data_gpu = torch.tensor(data, dtype=torch.float32).cuda()
    u_old_gpu = torch.tensor(u_old, dtype=torch.float32).cuda()

    # Normalizar y eliminar cualquier valor cero potencial
    u_old_gpu = normalize_columns(u_old_gpu)
    u_old_gpu = torch.fmax(u_old_gpu, torch.finfo(torch.float32).eps)

    # Elevar a la potencia de m
    um_gpu = u_old_gpu ** m

    # Calcular los centros de los clusters
    data_gpu = data_gpu.T  # Cambia la forma a (N, S)
    um_sum = um_gpu.sum(dim=1, keepdim=True)  # um_sum tiene forma (c, 1)
    cntr = (um_gpu @ data_gpu) / um_sum  # cntr tiene forma (c, N)
    
    # Calcular distancias usando la función calculate_distances adaptada a PyTorch
    d_gpu = calculate_distances(data_gpu, cntr, metric)
    d_gpu = torch.fmax(d_gpu, torch.finfo(torch.float32).eps)

    # Calcular jm
    jm = torch.sum(um_gpu * (d_gpu ** 2))

    # Actualizar la matriz de pertenencias u usando normalize_power_columns adaptada a PyTorch
    u = normalize_power_columns(d_gpu, -2. / (m - 1))

    # Pasar los resultados a la CPU si es necesario
    cntr = cntr.cpu().numpy()
    u = u.cpu().numpy()
    jm = jm.item()
    d = d_gpu.cpu().numpy()

    # Devolver los resultados
    return cntr, u, jm, d

def _fp_coeff(u):
    """
    Fuzzy partition coefficient fpc relative to fuzzy c-partitioned
    matrix u. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    # Convertir u a un tensor de PyTorch y moverlo a la GPU
    u_gpu = torch.tensor(u, dtype=torch.float32).cuda()
    n = u_gpu.shape[1]

    # Calcular el coeficiente de partición difuso en la GPU
    trace_u_ut = torch.trace(torch.matmul(u_gpu, u_gpu.T))
    fpc = trace_u_ut.item() / float(n)
    
    return fpc

def cmeans(data, c, m, error, maxiter, metric='euclidean', init=None, seed=None):
    start_time = time.time()
    
    # Convertir datos a tensor en la GPU
    data = torch.tensor(data, dtype=torch.float32).cuda()

    # Inicialización aleatoria de u0
    if init is None:
        if seed is not None:
            torch.manual_seed(seed)
        n = data.shape[1]
        u0 = torch.rand(c, n, dtype=torch.float32).cuda()
        u0 = normalize_columns(u0)
        init = u0.clone()
    else:
        u0 = torch.tensor(init, dtype=torch.float32).cuda()

    u = torch.fmax(u0, torch.finfo(torch.float32).eps)
    jm = torch.tensor([], dtype=torch.float32).cuda()
    p = 0

    while p < maxiter - 1:
        u2 = u.clone()
        cntr, u, Jjm, d = _cmeans0(data, u2, c, m, metric)
        jm = torch.cat((jm, Jjm.view(-1)))
        p += 1

        # Condición de parada
        if torch.norm(u - u2) < error:
            break

    error = torch.norm(u - u2)
    fpc = _fp_coeff(u)

    # Convertir a segundos
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Retornar resultados manteniendo en GPU
    return cntr, u, u0, d, jm, p, fpc


def fcm(resized_image, num_clusters, fila_interes, m=2.0, metric='euclidean'):
    """
    Aplicar Fuzzy C-Means clustering a una imagen redimensionada y devolver la imagen segmentada.

    Parameters
    ----------
    resized_image : torch.Tensor
        Imagen redimensionada como tensor de PyTorch.
    num_clusters : int
        Número de clusters para el algoritmo Fuzzy C-Means.
    m : float, optional
        Parámetro de fuzziness. Default es 2.0.
    metric : str, optional
        Métrica para el cálculo de distancias. Default es 'euclidean'.

    Returns
    -------
    torch.Tensor
        Imagen segmentada como tensor de PyTorch.
    """
    # Convertir la imagen a float32 y normalizar
    resized_image = resized_image.float() / 255.0
    
    # Reconfigurar la imagen al formato (S, N) en la GPU
    S, N = resized_image.shape[0] * resized_image.shape[1], resized_image.shape[2]
    data = resized_image.view(S, N)

    # Medir el tiempo de ejecución de la función cmeans
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, num_clusters, m, error=0.05, maxiter=20, metric=metric, init=None, seed=None)

    # Reconstruir la imagen segmentada
    cluster_membership = torch.argmax(u, dim=0)

    # Reshape del array cluster_membership para formar la imagen segmentada
    segmented_image = cluster_membership.view(resized_image.shape[0], resized_image.shape[1]).to(torch.uint8)

    # Normalizar la imagen segmentada
    max_val_gpu = torch.max(segmented_image)
    segmented_image_normalized = (segmented_image.float() * (255 / max_val_gpu)).to(torch.uint8)

    # Crop horizontal utilizando la función de utils
    _, image3 = utils_torch.crop_horizontal(segmented_image_normalized, fila_interes)

    # Calcular la frecuencia de cada cluster dentro del área de interés
    unique, counts = torch.unique(image3, return_counts=True)

    # Imprimir valores únicos y sus frecuencias
    # print("Unique clusters:", unique)
    # print("Cluster frequencies:", counts)

    # Encontrar el cluster con la mayor frecuencia
    max_cluster_idx = torch.argmax(counts)
    max_cluster = unique[max_cluster_idx]

    return segmented_image_normalized, max_cluster


def fcm2(resized_image, num_clusters, fila_interes, m=2.0, metric='euclidean'):
    """
    Aplicar Fuzzy C-Means clustering a una imagen redimensionada y devolver la imagen segmentada.

    Parameters
    ----------
    resized_image : torch.Tensor
        Imagen redimensionada como tensor de PyTorch.
    num_clusters : int
        Número de clusters para el algoritmo Fuzzy C-Means.
    m : float, optional
        Parámetro de fuzziness. Default es 2.0.
    metric : str, optional
        Métrica para el cálculo de distancias. Default es 'euclidean'.

    Returns
    -------
    torch.Tensor
        Imagen segmentada como tensor de PyTorch.
    """
    # Convertir la imagen a float32 y normalizar
    resized_image = resized_image.float() / 255.0
    
    # Reconfigurar la imagen al formato (S, N)
    S, N = resized_image.shape[0] * resized_image.shape[1], resized_image.shape[2]
    data = resized_image.view(S, N)

    # Medir el tiempo de ejecución de la función cmeans
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, num_clusters, m, error=0.05, maxiter=20, metric=metric, init=None, seed=None)

    # Reconstruir la imagen segmentada
    cluster_membership = torch.argmax(u, dim=0)

    # Reshape del array cluster_membership para formar la imagen segmentada
    segmented_image = cluster_membership.view(resized_image.shape[0], resized_image.shape[1]).to(torch.uint8)

    # Normalizar la imagen segmentada
    max_val_gpu = torch.max(segmented_image)
    segmented_image_normalized = (segmented_image.float() * (255 / max_val_gpu)).to(torch.uint8)

    # Recortar la imagen horizontalmente
    _, image3 = utils_torch.crop_horizontal(segmented_image_normalized, fila_interes)

    # Calcular la frecuencia de cada cluster dentro del área de interés
    unique, counts = torch.unique(image3, return_counts=True)

    # Imprimir valores únicos y sus frecuencias
    # print("Unique clusters:", unique)
    # print("Cluster frequencies:", counts)

    return segmented_image_normalized, image3
