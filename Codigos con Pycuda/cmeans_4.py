import cupy as cp
import numpy as np
import time
import cv2
from os.path import dirname, abspath, join
from normalize_columns2 import normalize_columns, normalize_power_columns

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
        float sum_um = 0.0f;

        // Initialize um with a very small value to avoid zero issues
        for (int i = 0; i < num_clusters; ++i) {
            u_old[data_idx * num_clusters + i] = fmaxf(u_old[data_idx * num_clusters + i], FLOAT_EPSILON);
            um[i] = powf(u_old[data_idx * num_clusters + i], m);
        }

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
mod = cp.RawModule(code=kernel_code)
cmeans_kernel = mod.get_function('cmeans_kernel')

def _cmeans0(data, u_old, c, m):
    """
    Single step in generic fuzzy c-means clustering algorithm using CuPy.
    """
    start_time = time.perf_counter()
    num_data, num_features = data.shape
    num_clusters = c
    
    # Normaliza las columnas de u_old
    u_old = normalize_columns(u_old)  # Asegúrate de tener esta función en cupy también

    # Preparar datos para CUDA
    d = cp.empty((num_data, num_clusters), dtype=cp.float32)
    cntr = cp.empty((num_clusters, num_features), dtype=cp.float32)
    jm = cp.empty(num_data, dtype=cp.float32)
    power = -2.0 / (m - 1)

    data_gpu = cp.asarray(data)
    u_old_gpu = cp.asarray(u_old)
    c_gpu = cp.asarray(np.array(c, dtype=np.float32))
    d_gpu = cp.asarray(d)
    cntr_gpu = cp.asarray(cntr)
    jm_gpu = cp.asarray(jm)

    # Define la configuración del kernel
    block_size = 256
    grid_size = (num_data + block_size - 1) // block_size

    # Ejecuta el kernel
    cmeans_kernel((grid_size,), (block_size,), (data_gpu, u_old_gpu, c_gpu, d_gpu, cntr_gpu, jm_gpu, cp.float32(power), cp.int32(num_data), cp.int32(num_features), cp.int32(num_clusters), cp.int32(m)))
    
    # Copia los resultados de vuelta a la CPU
    d_result = cp.asnumpy(d_gpu)
    cntr_result = cp.asnumpy(cntr_gpu)
    jm_result = cp.asnumpy(jm_gpu)

    jm_value = jm_result.sum()  # Total sum of jm across all data points
    
    end_time = time.perf_counter()

    # Verifica los resultados
    print("Cluster Centers (cntr):", cntr_result)
    print("Distances (d):", d_result)
    print("Objective Function (jm):", jm_value)
    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución hasta el primer resultado: {elapsed_time:.4f} segundos")

    # Visualizar los centros de los clústeres
    # Normaliza los centros de clústeres a rango [0, 255] y cambia la forma a imagen
    cntr_image = (cntr_result - cntr_result.min()) / (cntr_result.max() - cntr_result.min()) * 255
    cntr_image = np.uint8(cntr_image)
  
    for i in range(num_clusters):
        # Cada centro de clúster es una imagen de un solo píxel
        single_pixel_image = np.zeros((1, 1, 3), dtype=np.uint8)
        single_pixel_image[0, 0, :] = cntr_image[i, :]
        cv2.imshow(f'Cluster Center {i}', single_pixel_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cntr_result, d_result, jm_value

if __name__ == "__main__":
    # Parámetros de prueba
    num_clusters = 3
    m = 2.0

    # Cargar y procesar la imagen
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
