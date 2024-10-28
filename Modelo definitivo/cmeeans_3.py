import cupy as cp
import time
from normalize_columns2 import normalize_columns, normalize_power_columns
import detectar_horizonte2
import utils 
import cv2
from os.path import join, dirname, abspath

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
    
    jm = jm

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
    
    return fpc

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

    
    return cntr, u, u0, d, jm, p, fpc

def fcm(resized_image, num_clusters, m=2.0, metric='euclidean'):
    """
    Cargar la imagen, aplicar Fuzzy C-Means clustering y devolver la imagen segmentada.

    Parameters
    ----------
    image_path : str
        Ruta del archivo de imagen a cargar.
    num_clusters : int
        Número de clusters para el algoritmo Fuzzy C-Means.
    m : float, optional
        Parámetro de fuzziness. Default es 2.0.
    metric : str, optional
        Métrica para el cálculo de distancias. Default es 'euclidean'.

    Returns
    -------
    numpy.ndarray
        Imagen segmentada como un array de NumPy.
    """
    # Cargar y redimensionar la imagen
    # Convertir la imagen a float32 y normalizar
    resized_image = cp.array(resized_image, dtype=cp.float32)
    # Normalizar la imagen dividiéndola por 255.0
    resized_image /= 255.0
    
    # Reconfigurar la imagen al formato (S, N) en la GPU
    S, N = resized_image.shape[0] * resized_image.shape[1], resized_image.shape[2]
    data = resized_image.reshape(S, N)

    # Medir el tiempo de ejecución de la función cmeans
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, num_clusters, m, error=0.05, maxiter=10, metric=metric, init=None, seed=None)
    u = cp.asarray(u)
    # Reconstruir la imagen segmentada
    cluster_membership = cp.argmax(u, axis=0)
    # Supongamos que cluster_membership es un array de CuPy
# Reshape del array cluster_membership para formar la imagen segmentada
    segmented_image = cp.reshape(cluster_membership, (resized_image.shape[0], resized_image.shape[1])).astype(cp.uint8)

    # Normalizar la imagen segmentada
    max_val_gpu = cp.max(segmented_image)
    segmented_image_normalized = (segmented_image * (255 / max_val_gpu)).astype(cp.uint8)

    # Ejemplo de generación de un array normalizado (si es necesario)
    #segmented_image_normalized = cp.random.rand(100, 100)  # Ejemplo de array normalizado
    segmented_image_normalized = (segmented_image_normalized * 255).astype(cp.uint8)
    segmented_image_normalized_np = cp.asnumpy(segmented_image_normalized)
    cv2.imshow("segmentado", segmented_image_normalized_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Paso 5: Calcular la frecuencia de cada cluster

    fila_interes, imagen = detectar_horizonte2.find_horizontal_line(resized_image)
    # Recortar la imagen horizontalmente (supongamos que crop_horizontal también trabaja con CuPy)
    _, image3 = utils.crop_horizontal(segmented_image_normalized, fila_interes)
    image3_np=cp.asnumpy(image3)
    cv2.imshow("cortado", image3_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Calcular la frecuencia de cada cluster dentro del área de interés
    unique, counts = cp.unique(image3, return_counts=True)

    # Imprimir valores únicos y sus frecuencias
    print("Unique clusters:", unique)
    print("Cluster frequencies:", counts)

    # Encontrar el cluster con la mayor frecuencia
    """max_cluster_idx = cp.argmin(counts)
    max_cluster = unique[max_cluster_idx]
    mask_max_cluster = cp.zeros_like(image3, dtype=cp.uint8)
    mask_max_cluster[image3 == max_cluster] = 255  # Asignar blanco a los píxeles del cluster menos frecuentes"""
    two_min_clusters_idx = cp.argsort(counts)[:2]  # Ordena y toma los dos primeros índices

    # Obtener los valores de los dos clústeres más pequeños
    two_min_clusters = unique[two_min_clusters_idx]

    # Crear la máscara vacía
    mask_min_clusters = cp.zeros_like(image3, dtype=cp.uint8)

    # Hacer blancos (255) los píxeles que pertenecen a cualquiera de los dos clústeres
    mask_min_clusters[cp.isin(image3, two_min_clusters)] = 255

    # Convertir a NumPy para visualizar con OpenCV
    mask_max_cluster_cpu = cp.asnumpy(mask_min_clusters)

    # Mostrar la imagen utilizando OpenCV
    cv2.imshow("original_cmeans", mask_max_cluster_cpu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Liberar memoria de GPU al final del script
    cp.get_default_memory_pool().free_all_blocks()

    return mask_max_cluster_cpu, fila_interes

def fcm2(resized_image, num_clusters, m=2.0, metric='euclidean'):
    """
    Cargar la imagen, aplicar Fuzzy C-Means clustering y devolver la imagen segmentada.

    Parameters
    ----------
    image_path : str
        Ruta del archivo de imagen a cargar.
    num_clusters : int
        Número de clusters para el algoritmo Fuzzy C-Means.
    m : float, optional
        Parámetro de fuzziness. Default es 2.0.
    metric : str, optional
        Métrica para el cálculo de distancias. Default es 'euclidean'.

    Returns
    -------
    numpy.ndarray
        Imagen segmentada como un array de NumPy.
    """
    # Cargar y redimensionar la imagen
    # Convertir la imagen a float32 y normalizar
    resized_image = cp.array(resized_image, dtype=cp.float32)
    # Normalizar la imagen dividiéndola por 255.0
    resized_image /= 255.0
    
    # Reconfigurar la imagen al formato (S, N) en la GPU
    S, N = resized_image.shape[0] * resized_image.shape[1], resized_image.shape[2]
    data = resized_image.reshape(S, N)

    # Medir el tiempo de ejecución de la función cmeans
    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, num_clusters, m, error=0.05, maxiter=10, metric=metric, init=None, seed=None)
    u = cp.asarray(u)
    # Reconstruir la imagen segmentada
    cluster_membership = cp.argmax(u, axis=0)
    # Supongamos que cluster_membership es un array de CuPy
# Reshape del array cluster_membership para formar la imagen segmentada
    segmented_image = cp.reshape(cluster_membership, (resized_image.shape[0], resized_image.shape[1])).astype(cp.uint8)

    # Normalizar la imagen segmentada
    max_val_gpu = cp.max(segmented_image)
    segmented_image_normalized = (segmented_image * (255 / max_val_gpu)).astype(cp.uint8)

    # Ejemplo de generación de un array normalizado (si es necesario)
    #segmented_image_normalized = cp.random.rand(100, 100)  # Ejemplo de array normalizado
    segmented_image_normalized = (segmented_image_normalized * 255).astype(cp.uint8)
    segmented_image_normalized_np = cp.asnumpy(segmented_image_normalized)
    cv2.imshow("segmentado", segmented_image_normalized_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Paso 5: Calcular la frecuencia de cada cluster

    fila_interes, imagen = detectar_horizonte2.find_horizontal_line(resized_image)
    # Recortar la imagen horizontalmente (supongamos que crop_horizontal también trabaja con CuPy)
    _, image3 = utils.crop_horizontal(segmented_image_normalized, fila_interes)
    image3_np=cp.asnumpy(image3)
    cv2.imshow("cortado", image3_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Calcular la frecuencia de cada cluster dentro del área de interés
    unique, counts = cp.unique(image3, return_counts=True)

    # Imprimir valores únicos y sus frecuencias
    print("Unique clusters:", unique)
    print("Cluster frequencies:", counts)

    # Encontrar el cluster con la mayor frecuencia
    """max_cluster_idx = cp.argmin(counts)
    max_cluster = unique[max_cluster_idx]
    mask_max_cluster = cp.zeros_like(image3, dtype=cp.uint8)
    mask_max_cluster[image3 == max_cluster] = 255  # Asignar blanco a los píxeles del cluster menos frecuentes"""
    two_min_clusters_idx = cp.argsort(counts)[:1]  # Ordena y toma los dos primeros índices

    # Obtener los valores de los dos clústeres más pequeños
    two_min_clusters = unique[two_min_clusters_idx]

    # Crear la máscara vacía
    mask_min_clusters = cp.zeros_like(image3, dtype=cp.uint8)

    # Hacer blancos (255) los píxeles que pertenecen a cualquiera de los dos clústeres
    mask_min_clusters[cp.isin(image3, two_min_clusters)] = 255

    # Convertir a NumPy para visualizar con OpenCV
    mask_max_cluster_cpu = cp.asnumpy(mask_min_clusters)

    # Mostrar la imagen utilizando OpenCV
    cv2.imshow("original_cmeans", mask_max_cluster_cpu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Liberar memoria de GPU al final del script
    cp.get_default_memory_pool().free_all_blocks()

    return mask_max_cluster_cpu, fila_interes

# Ejemplo de uso
if __name__ == "__main__":
    filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
    image = cv2.imread(filename)
    image_cupy = cp.array(image, dtype=cp.float32)
    new_width = 200
    orig_height, orig_width, channels = image_cupy.shape
    new_height = int(orig_height * new_width / orig_width)
    resized_image = utils.resize_image_bgr(image_cupy, (new_height, new_width))
    segmented_image = fcm(resized_image, num_clusters=4)
    
    print(segmented_image)