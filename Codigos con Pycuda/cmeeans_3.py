
from os.path import dirname, abspath, join
from sys import argv
import numpy as np
import cupy as cp
import cv2
import time

def normalize_columns(u):
    return u / u.sum(axis=0, keepdims=True)

def normalize_power_columns(d, exp):
    return (d ** exp) / cp.sum(d ** exp, axis=0, keepdims=True)

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

    # Normalizing, then eliminating any potential zero values.
    u_old_gpu = normalize_columns(u_old_gpu)
    u_old_gpu = cp.fmax(u_old_gpu, cp.finfo(cp.float32).eps)

    um_gpu = u_old_gpu ** m

    # Calculate cluster centers
    data_gpu = data_gpu.T  # Now data_gpu has shape (N, S)
    um_sum = cp.atleast_2d(um_gpu.sum(axis=1)).T  # um_sum has shape (c, 1)
    cntr_gpu = (um_gpu @ data_gpu.T) / um_sum  # cntr_gpu has shape (c, N)

    d_gpu = calculate_distances(data_gpu.T, cntr_gpu, metric)  # data_gpu.T has shape (S, N), cntr_gpu has shape (c, N)
    d_gpu = cp.fmax(d_gpu, cp.finfo(cp.float32).eps)

    jm = cp.sum(um_gpu * (d_gpu ** 2))

    u_gpu = normalize_power_columns(d_gpu, -2. / (m - 1))

    # Move results back to CPU
    cntr = cp.asnumpy(cntr_gpu)
    u = cp.asnumpy(u_gpu)
    jm = jm.get()
    d = cp.asnumpy(d_gpu)

    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()

    return cntr, u, jm, d

def main(image_path, num_clusters=3, m=2.0, metric='euclidean'):
    start_time = time.perf_counter()
    # Load and resize the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    scale_factor = 200.0 / height
    new_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (new_width, 200))

    # Convert image to float32 and normalize
    data = resized_image.astype(np.float32) / 255.0

    # Reshape the image to the format (S, N)
    S, N = data.shape[0] * data.shape[1], data.shape[2]
    data = data.reshape(S, N)

    # Initialize u_old randomly
    u_old = np.random.rand(num_clusters, S)
    u_old = normalize_columns(u_old)

    # Run the fuzzy c-means algorithm
    cntr, u, jm, d = _cmeans0(data, u_old, num_clusters, m, metric)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución hasta el primer resultado: {elapsed_time:.4f} segundos")
    # Print results
    print("Cluster Centers:\n", cntr)
    print("Final Membership Matrix:\n", u)
    print("Objective Function Value:\n", jm)
    print("Distance Matrix:\n", d)

if __name__ == "__main__":
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
    image_path=filename
    main(image_path)
    
