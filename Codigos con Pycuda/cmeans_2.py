import cv2
import pycuda.driver as cuda
import pycuda.compiler as compiler
import numpy as np
from os.path import dirname, abspath, join
from sys import argv
from normalize_columns import normalize_columns, normalize_power_columns
import time
import cupy as cp

# Define el código CUDA como una cadena de texto
kernel_code = """
#define FLOAT_EPSILON 1e-8

__device__ float distance(float *data, float *cntr, int data_idx, int cluster_idx, int num_features) {
    float dist = 0.0f;
    for (int i = 0; i < num_features; ++i) {
        float diff = data[data_idx * num_features + i] - cntr[cluster_idx * num_features + i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

__global__ void cmeans_kernel(float *data, float *u_old, float *c, float *d, float *cntr, float *jm, float power, int num_data, int num_features, int num_clusters, int m) {
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_idx < num_data) {
        float um[1024];  // Assuming num_clusters <= 1024

        // Initialize um with a very small value to avoid zero issues
        for (int i = 0; i < num_clusters; ++i) {
            u_old[data_idx * num_clusters + i] = fmaxf(u_old[data_idx * num_clusters + i], FLOAT_EPSILON);
            um[i] = powf(u_old[data_idx * num_clusters + i], m);
        }

        float sum_um = 0.0f;
        for (int i = 0; i < num_clusters; ++i) {
            sum_um += um[i];
        }

        if (sum_um > 0) {
            for (int j = 0; j < num_features; ++j) {
                float sum_data = 0.0f;
                for (int i = 0; i < num_clusters; ++i) {
                    sum_data += um[i] * data[data_idx * num_features + j];
                }
                cntr[data_idx * num_features + j] = sum_data / sum_um;
            }
        }

        // Calculate distances using the device function
        for (int i = 0; i < num_clusters; ++i) {
            d[data_idx * num_clusters + i] = distance(data, cntr, data_idx, i, num_features);
        }

        // Calculate jm
        float sum_jm = 0.0f;
        for (int i = 0; i < num_clusters; ++i) {
            sum_jm += um[i] * d[data_idx * num_clusters + i] * d[data_idx * num_clusters + i];
        }
        jm[data_idx] = sum_jm;

        // Normalize d
        float sum_d = 0.0f;
        for (int i = 0; i < num_clusters; ++i) {
            sum_d += powf(d[data_idx * num_clusters + i], power);
        }
        if (sum_d > 0) {
            for (int i = 0; i < num_clusters; ++i) {
                d[data_idx * num_clusters + i] /= powf(sum_d, 1.0f / power);
            }
        }
    }
}
"""

# Compila el código CUDA
mod = compiler.SourceModule(kernel_code)

# Obtén las funciones del módulo compilado
cmeans_kernel = mod.get_function("cmeans_kernel")

def _cmeans0(data, u_old, c, m):
    """
    Single step in generic fuzzy c-means clustering algorithm using PyCUDA.
    """
    start_time = time.perf_counter()
    num_data, num_features = data.shape
    print(num_features)
    num_clusters = c
    
    # Normaliza las columnas de u_old
    u_old = normalize_columns(u_old)

    # Preparar datos para CUDA
    d = np.empty((num_data, num_clusters), dtype=np.float32)
    cntr = np.empty((num_clusters, num_features), dtype=np.float32)
    jm = np.empty(num_data, dtype=np.float32)
    power = -2.0 / (m - 1)

    data_gpu = cuda.mem_alloc(data.nbytes)
    u_old_gpu = cuda.mem_alloc(u_old.nbytes)
    c_gpu = cuda.mem_alloc(np.array(c, dtype=np.float32).nbytes)
    d_gpu = cuda.mem_alloc(d.nbytes)
    cntr_gpu = cuda.mem_alloc(cntr.nbytes)
    jm_gpu = cuda.mem_alloc(jm.nbytes)

    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(u_old_gpu, u_old)
    cuda.memcpy_htod(c_gpu, np.array(c, dtype=np.float32))

    # Define la configuración del kernel
    block_size = 256
    grid_size = (num_data + block_size - 1) // block_size
    cmeans_kernel(data_gpu, u_old_gpu, c_gpu, d_gpu, cntr_gpu, jm_gpu, np.float32(power), np.int32(num_data), np.int32(num_features), np.int32(num_clusters), np.int32(m), block=(block_size, 1, 1), grid=(grid_size, 1))
    
    cuda.memcpy_dtoh(d, d_gpu)
    cuda.memcpy_dtoh(cntr, cntr_gpu)
    cuda.memcpy_dtoh(jm, jm_gpu)

    jm_value = jm.sum()  # Total sum of jm across all data points
    end_time = time.perf_counter()

    # Verifica los resultados
    print("Cluster Centers (cntr):", cntr)
    print("Distances (d):", d)
    print("Objective Function (jm):", jm_value)
    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución hasta el primer resultado: {elapsed_time:.4f} segundos")

    # Visualizar los centros de los clústeres
    # Normaliza los centros de clústeres a rango [0, 255] y cambia la forma a imagen
    cntr_image = (cntr - cntr.min()) / (cntr.max() - cntr.min()) * 255
    cntr_image = np.uint8(cntr_image)
  
    
    for i in range(num_clusters):
        # Cada centro de clúster es una imagen de un solo píxel
        single_pixel_image = np.zeros((1, 1, 3), dtype=np.uint8)
        single_pixel_image[0, 0, :] = cntr_image[i, :]
        cv2.imshow(f'Cluster Center {i}', single_pixel_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Libera la memoria del dispositivo
    data_gpu.free()
    u_old_gpu.free()
    c_gpu.free()
    d_gpu.free()
    cntr_gpu.free()
    jm_gpu.free()

    return cntr, d, jm_value

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
    cntr, d, jm_value = _cmeans0(data, u_old, num_clusters, m)

    # Imprime resultados
    print("Cluster Centers (cntr):")
    print(cntr)
    print("\nDistances (d):")
    print(d)
    print("\nObjective Function (jm):")
    print(jm_value)
