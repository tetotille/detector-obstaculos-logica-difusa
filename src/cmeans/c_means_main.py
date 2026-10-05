import numpy as cp
import time 
import cv2
from src.utils import utils, block_framed, neighbor_framed, neighbor_framed_np
from src.detector_horizonte import detectar_horizonte2
from os.path import join, dirname, abspath
from scipy.ndimage import label

def normalize_power_columns(matrix, power):
    powered_matrix = cp.power(matrix, power)
    column_sums = cp.sum(powered_matrix, axis=0)
    # Evitar división por cero
    column_sums[column_sums == 0] = 1e-10
    normalized_matrix = powered_matrix / column_sums
    return normalized_matrix

def normalize_columns(u):
    column_sums = cp.sum(u, axis=0)
    column_sums[column_sums == 0] = 1e-10
    normalized_u = u / column_sums
    return normalized_u

def calculate_distances(data, centers, metric='euclidean'):
    if metric != 'euclidean':
        raise NotImplementedError(f"Metric '{metric}' not implemented.")
    N, Q = data.shape
    C = centers.shape[0]

    data_expanded = cp.expand_dims(data, 1)  # Shape (N, 1, Q)
    centers_expanded = cp.expand_dims(centers, 0)  # Shape (1, C, Q)
    diff = data_expanded - centers_expanded  # Shape (N, C, Q)
    
    sq_dists = cp.sum(diff ** 2, axis=2)  # Shape (N, C)
    return cp.transpose(sq_dists)

def _cmeans0(data, u_old, c, m, metric='euclidean'):
    data_f = cp.array(data, dtype=cp.float32)
    u_old_f = cp.array(u_old, dtype=cp.float32)
    
    u_old_f = normalize_columns(u_old_f)
    u_old_f = cp.fmax(u_old_f, cp.finfo(cp.float32).eps)

    um = u_old_f ** m

    data_f = data_f.T  # Now data_f has shape (N, S)
    um_sum = cp.atleast_2d(um.sum(axis=1)).T
    # Evitar división por cero en centros
    um_sum[um_sum == 0] = 1e-10
    cntr = (um @ data_f) / um_sum
    
    d_f = calculate_distances(data_f, cntr, metric)
    d = cp.fmax(d_f, cp.finfo(cp.float32).eps)

    jm = cp.sum(um * (d_f ** 2))
    u = normalize_power_columns(d_f, -2. / (m - 1))

    return cntr, u, jm, d

def _fp_coeff(u):
    u_f = cp.array(u, dtype=cp.float32)
    n = u_f.shape[1]
    trace_u_ut = cp.trace(cp.dot(u_f, u_f.T))
    fpc = trace_u_ut / float(n)
    return fpc

def cmeans(data, c, m, error, maxiter, metric='euclidean', init=None, seed=None):
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

    return cntr, u, u0, d, jm, p, fpc

def fcm(resized_image, num_clusters, m=2.0, metric='euclidean',show_images=False,punto_horizonte=0):
    resized_image = cp.array(resized_image, dtype=cp.float32)
    resized_image /= 255.0
    
    S, N = resized_image.shape[0] * resized_image.shape[1], resized_image.shape[2]
    data = resized_image.reshape(S, N)

    cntr, u, u0, d, jm, p, fpc = cmeans(data.T, num_clusters, m, error=0.05, maxiter=10, metric=metric, init=None, seed=None)
    u = cp.asarray(u)
    cluster_membership = cp.argmax(u, axis=0)
    segmented_image = cp.reshape(cluster_membership, (resized_image.shape[0], resized_image.shape[1])).astype(cp.uint8)

    max_val = cp.max(segmented_image)
    if max_val == 0: max_val = 1
    segmented_image_normalized = (segmented_image * (255 / max_val)).astype(cp.uint8)

    _, image3 = utils.crop_horizontal(segmented_image_normalized, punto_horizonte)
    
    unique, counts = cp.unique(image3, return_counts=True)
    if len(counts) > 0:
        max_cluster_idx = cp.argmin(counts)
        max_cluster = unique[max_cluster_idx]
    else:
        max_cluster = 0

    mask_max_cluster = cp.zeros_like(image3, dtype=cp.uint8)
    mask_max_cluster[image3 == max_cluster] = 255 

    cuadros = neighbor_framed_np(mask_max_cluster)
    return mask_max_cluster, punto_horizonte, cuadros

if __name__ == "__main__":
    filename = join(dirname(dirname(dirname(abspath(__file__)))), "assets/images/barco.jpg")
    image = cv2.imread(filename)
    if image is not None:
        resized_image = cv2.resize(image, (256, 192))
        segmented_image = fcm(resized_image, num_clusters=4)
        print(segmented_image[2])
