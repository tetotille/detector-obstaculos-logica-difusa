from os.path import dirname, abspath, join
from sys import argv
import numpy as np
import cupy as cp
import cv2
import time
from normalize_columns2 import normalize_columns, normalize_power_columns

def calculate_distances(data, centers, metric='euclidean'):
    """
    Calculate distances between data points and cluster centers using Cupy.

    Parameters
    ----------
    data : cupy.ndarray
        Data to be analyzed. Shape is (N, Q) where N is the number of data points and Q is the number of features.
    centers : cupy.ndarray
        Cluster centers. Shape is (C, Q) where C is the number of clusters and Q is the number of features.
    metric : str
        Metric to use. Currently only 'euclidean' is supported.

    Returns
    -------
    cupy.ndarray
        Distance matrix. Shape is (C, N) where C is the number of clusters and N is the number of data points.
    """
    if metric != 'euclidean':
        raise NotImplementedError(f"Metric '{metric}' not implemented.")
    # Obtener las dimensiones
    N, Q = data.shape
    C = centers.shape[0]

    # Calculate squared differences
    data_expanded = cp.expand_dims(data, 1)  # Shape (N, 1, Q)
    centers_expanded = cp.expand_dims(centers, 0)  # Shape (1, C, Q)
    diff = data_expanded - centers_expanded  # Shape (N, C, Q)
    
    # Calculate squared Euclidean distances
    sq_dists = cp.sum(diff ** 2, axis=2)  # Shape (N, C)
    
    # Return the transpose to match the expected output shape (C, N)
    return cp.transpose(sq_dists)

def _cmeans0(data, u_old, c, m, metric='euclidean'):
    # Move data to GPU
    data_gpu = cp.array(data, dtype=cp.float32)
    u_old_gpu = cp.array(u_old, dtype=cp.float32)
    
    # Create events for timing
    

    # Normalizing, then eliminating any potential zero values.
    u_old_gpu = normalize_columns(u_old_gpu)
    u_old_gpu = cp.fmax(u_old_gpu, cp.finfo(cp.float32).eps)

    um_gpu = u_old_gpu ** m

    # Calculate cluster centers
    data_gpu = data_gpu.T  # Now data_gpu has shape (N, S)
    um_sum = cp.atleast_2d(um_gpu.sum(axis=1)).T  # um_sum has shape (c, 1)
    cntr = (um_gpu @ data_gpu) / um_sum  # cntr_gpu has shape (c, N)
    
    d_gpu = calculate_distances(data_gpu, cntr, metric)  # data_gpu.T has shape (S, N), cntr_gpu has shape (c, N)
    d = cp.fmax(d_gpu, cp.finfo(cp.float32).eps)

    jm = cp.sum(um_gpu * (d_gpu ** 2))

    u = normalize_power_columns(d_gpu, -2. / (m - 1))

    # Move results back to CPU
    jm = jm.get()
    
    

    # Free GPU memory

    
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
    u_gpu = cp.array(u, dtype=cp.float32)
    n = u_gpu.shape[1]

    # Compute the fuzzy partition coefficient on GPU
    trace_u_ut = cp.trace(cp.dot(u_gpu, u_gpu.T))
    fpc = trace_u_ut / float(n)
    
    return fpc.get()

def cmeans(data, c, m, error, maxiter, metric='euclidean', init=None, seed=None):
    start_time = time.time()
    if init is None:
        if seed is not None:
            cp.random.seed(seed=seed)
        n = data.shape[1]
        u0 = cp.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    else:
        u0 = cp.array(init)
    u = cp.fmax(u0, cp.finfo(cp.float32).eps)

    jm = cp.zeros(0)
    p = 0

    while p < maxiter - 1:
        u2 = u.copy()
        cntr, u, Jjm, d = _cmeans0(data, u2, c, m, metric)
        jm = cp.hstack((jm, Jjm))
        p += 1

        if cp.linalg.norm(u - u2) < error:
            break

    error = cp.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

 # Convert to seconds
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tiempo total de ejecución en GPU: {elapsed_time:.6f} segundos")
    
    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    
    return cntr, cp.asnumpy(u), cp.asnumpy(u0), cp.asnumpy(d), cp.asnumpy(jm), p, fpc

def reconstruct_segmented_image(u, image_np):
    # Paso 1: Obtener el índice del cluster más probable para cada píxel
    cluster_membership = cp.argmax(u, axis=0)

    # Paso 2: Reconstruir la imagen segmentada
    segmented_image = cp.reshape(cluster_membership, image_np).astype(cp.uint8)
    
    # Paso 3: Normalizar la imagen segmentada
    max_val = cp.max(segmented_image)
    segmented_image_normalized = (segmented_image * (255 / max_val)).astype(cp.uint8)

    # Convertir el array de CuPy a NumPy
    segmented_image_normalized_np = cp.asnumpy(segmented_image_normalized)
    
    # Paso 4: Usar OpenCV para mostrar o guardar la imagen
    cv2.imshow('Segmented Image', segmented_image_normalized_np)
    #cv2.imwrite('segmented_image.png', segmented_image_normalized_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path, num_clusters=3, m=2.0, metric='euclidean'):
    # Cargar y redimensionar la imagen
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    scale_factor = 200.0 / width
    new_height = int(height * scale_factor)
    print(new_height)
    resized_image = cv2.resize(image, (200, new_height))
    # Mostrar la imagen inicial
    cv2.imshow('Imagen inicial', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convertir la imagen a float32 y normalizar
    data = resized_image.astype(np.float32) / 255.0

    # Reconfigurar la imagen al formato (S, N)
    S, N = data.shape[0] * data.shape[1], data.shape[2]
    data = data.reshape(S, N)

    # Medir el tiempo de ejecución de la función cmeans
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, num_clusters, m, error=0.00005, maxiter=10, metric=metric, init=None, seed=None)

    # Reconstruir y mostrar la imagen segmentada
    u_cp = cp.asarray(u)
    reconstruct_segmented_image(u_cp, resized_image.shape[:2])

    # Imprimir resultados
    print("Cluster Centers:\n", cntr)
    print("Final Membership Matrix:\n", u)
    print("Objective Function Value:\n", jm)
    print("Distance Matrix:\n", d)
    # Liberar memoria de GPU al final del script
    cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
    image_path = filename
    # Imprimir formas
    main(image_path)