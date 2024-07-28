import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
import numpy as np
from os.path import dirname, abspath, join
from sys import argv
from normalize_columns import normalize_columns, normalize_power_columns
import time

# Define el código CUDA como una cadena de texto
kernel_code = """
__device__ float distance(float *data, float *cntr, int data_idx, int cluster_idx, int num_features) {
    float dist = 0.0f;
    for (int i = 0; i < num_features; ++i) {
        float diff = data[data_idx * num_features + i] - cntr[cluster_idx * num_features + i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

__global__ void kernel(
    float *u_old, float *um, float *data, float *data2, 
    float *d_T, float *d_sums, float *cntr, float *d, 
    float *jm, int num_data, int num_features, int num_clusters, 
    int m, int num_rows, int num_cols, int num_elements, float EPSILON
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Definición de variables de uso local
    float um_val = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    // Corrección en el manejo de u_old y EPSILON
    if (idx < num_data) 
    {
        if (u_old[idx] < EPSILON) 
        {
            u_old[idx] = EPSILON;
        }
        um[idx] = powf(u_old[idx], m);
    }

    // Corrección en el manejo de data2
    if (row < num_rows && col < num_cols) 
    {
        int input_idx = row * num_cols + col;
        int output_idx = col * num_rows + row;
        data2[output_idx] = data[input_idx];
    }

    // Corrección en el cálculo de distancias y actualización de cntr
    if (idx < num_clusters) {
        sum1 = 0.0f;
        sum2 = 0.0f;
        for (int feature_idx = 0; feature_idx < num_features; ++feature_idx) 
        {
            um_val = um[idx * num_features + feature_idx];
            float T_val = d_T[feature_idx * num_data + idx];
            sum2 += um_val;
            sum1 += um_val * T_val;
        }
        d_sums[idx] = sum2;
        if (sum2 > 0) { // Asegúrate de que `d_sums[idx]` no sea cero
            cntr[idx * num_features + idx] = sum1 / d_sums[idx];
        }
        for (int data_idx = 0; data_idx < num_data; ++data_idx) {
            float dist = distance(data2, cntr, data_idx, idx, num_features);
            d[data_idx * num_clusters + idx] = dist;
        }
    }

    // Corrección en el cálculo de d
    if (idx < num_data) {
        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
            d[idx * num_clusters + cluster_idx] = fmaxf(d[idx * num_clusters + cluster_idx], EPSILON);
            d[idx * num_clusters + cluster_idx] = d[idx * num_clusters + cluster_idx] * d[idx * num_clusters + cluster_idx];
        }
    }

    // Manejo de `shared_sum` y la reducción
    __shared__ float shared_sum[256]; // Ajusta el tamaño del arreglo según sea necesario

    if (idx < num_clusters * num_elements) {
        int cluster_idx = idx / num_elements;
        int element_idx = idx % num_elements;
        float value = um[cluster_idx * num_elements + element_idx] * d[element_idx];
        shared_sum[tid] = value;
    } 
    else 
    {
        shared_sum[tid] = 0.0f;
    }

    __syncthreads();

    // Reducción para sumar los elementos en el bloque
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(jm, shared_sum[0]);
    }
}        
"""

# Compila el código CUDA
mod = compiler.SourceModule(kernel_code)

# Obtén las funciones del módulo compilado
cmeans_kernel = mod.get_function("kernel")

def _cmeans0(data, u_old, c, m):

     # Medir el tiempo de ejecución
    start_time = time.perf_counter()
    """
    Single step in generic fuzzy c-means clustering algorithm using PyCUDA.
    """
    num_data, num_features = data.shape
    num_clusters = c
    data2 = np.zeros((num_data, num_features), dtype=np.float32)
    u_old = normalize_columns(u_old)
    jm = np.zeros(num_data, dtype=np.float32)
    d = np.empty((num_data, num_clusters), dtype=np.float32)
    cntr = np.empty((num_clusters, num_features), dtype=np.float32)
    um = np.zeros_like(u_old, dtype=np.float32)
    d_T = np.zeros((num_features, num_data), dtype=np.float32)
    d_sums = np.zeros(num_clusters, dtype=np.float32)

   # Preparar datos para CUDA
    data2_gpu = cuda.mem_alloc(data2.nbytes)
    data_gpu = cuda.mem_alloc(data.nbytes)
    u_old_gpu = cuda.mem_alloc(u_old.nbytes)
    um_gpu = cuda.mem_alloc(um.nbytes)
    d_T_gpu = cuda.mem_alloc(d_T.nbytes)
    d_sums_gpu = cuda.mem_alloc(d_sums.nbytes)
    d_gpu = cuda.mem_alloc(d.nbytes)
    cntr_gpu = cuda.mem_alloc(cntr.nbytes)
    jm_gpu = cuda.mem_alloc(jm.nbytes)
    power=(-2.)/(m-1)

    num_rows, num_cols = data2.shape
    num_elements = num_rows * num_cols
    EPSILON = 1e-10


    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(u_old_gpu, u_old)

    # Define la configuración del kernel
    block_size = 256
    grid_size = (num_data + block_size - 1) // block_size
    cmeans_kernel(
        u_old_gpu, um_gpu, data_gpu, data2_gpu, d_T_gpu, d_sums_gpu, cntr_gpu, d_gpu, jm_gpu, np.int32(num_data), 
        np.int32(num_features), np.int32(num_clusters), np.int32(m), np.int32(num_rows), np.int32(num_cols), 
        np.int32(num_elements), np.float32(EPSILON), block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    cuda.memcpy_dtoh(d, d_gpu)
    cuda.memcpy_dtoh(cntr, cntr_gpu)
    cuda.memcpy_dtoh(jm, jm_gpu)

    jm_value = jm.sum()  # Total sum of jm across all data points
    u = normalize_power_columns(d, power)

     # Medir el tiempo hasta el primer resultado
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
    del data_gpu
    del u_old_gpu
    del c_gpu
    del d_gpu
    del cntr_gpu
    del jm_gpu

    cuda.Context.get_current().pop()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cntr, d, jm_value, u

if __name__ == "__main__":
    # Parámetros de prueba
    c = 3
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
    data = np.reshape(pixels, (-1, 3)).T
    num_data = data.shape[1]
    
    # Generar u_old con la forma correcta
    u_old = np.random.rand(num_data, c).astype(np.float32)

    # Ejecuta la función _cmeans0
    cntr, d, jm_value, u = _cmeans0(data, u_old, c, m)

    # Imprime resultados
    print("Cluster Centers (cntr):")
    print(cntr)
    print("\nDistances (d):")
    print(d)
    print("\nObjective Function (jm):")
    print(jm_value)

