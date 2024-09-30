import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from os.path import dirname, abspath, join
from sys import argv

# Kernel en CUDA para normalizar columnas
kernel_code = """
__global__ void normalize_columns(float *columns, float *column_sums, float *normalized_columns, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        int idx = row * cols + col;
        normalized_columns[idx] = columns[idx] / column_sums[col];
    }
}
#include <math.h>
extern "C" __global__ void normalize_power_columns_kernel(float *x, float *result, int rows, int cols, float exponent, float eps) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < cols && row < rows) {
            // Calculate index
            int idx = row + col * rows;

            // Load and normalize the value
            float value = x[idx];

            // Normalize columns
            if (exponent < 0) {
                value = value / eps;
                value = powf(value, exponent);
            } else {
                value = powf(value, exponent);
            }

            // Store the result
            result[idx] = value;
        }
    }
"""

# Compilamos el código CUDA
mod = SourceModule(kernel_code)

def normalize_power_columns(x, exponent):

    assert np.all(x >= 0.0)
    
    # Normalize x
    x = x.astype(np.float32)
    x_max = np.max(x, axis=0, keepdims=True)
    eps = np.finfo(x.dtype).eps
    x = x / x_max
    x = np.fmax(x, eps)
    normalize_power_columns_cuda = mod.get_function("normalize_power_columns_kernel")
    rows, cols = (x).shape
    
    result = np.zeros_like(x)
    
    # Allocate memory on the device
    x_gpu = cuda.mem_alloc(x.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)

    # Copy data to the device
    cuda.memcpy_htod(x_gpu, x)
    
    # Launch the CUDA kernel
    block_dim = (16, 16, 1)
    grid_dim = ((cols + block_dim[0] - 1) // block_dim[0], (rows + block_dim[1] - 1) // block_dim[1])
    normalize_power_columns_cuda(x_gpu, result_gpu, np.int32(rows), np.int32(cols), np.float32(exponent), np.float32(eps), block=block_dim, grid=grid_dim)
    # Copy result back to the host
    cuda.memcpy_dtoh(result, result_gpu)

    result=normalize_columns(result)
    
    # Normalize the columns

    return result
    

def normalize_columns(columns):
    mod = SourceModule(kernel_code)
    normalize_columns_kernel = mod.get_function("normalize_columns")

    
    rows, cols = (columns).shape

    
    columns = columns.astype(np.float32)

    # Creamos un array para almacenar las sumas de las columnas
    column_sums = np.sum(columns, axis=0).astype(np.float32)

    # Reservamos memoria en la GPU
    columns_gpu = cuda.mem_alloc(columns.nbytes)
    column_sums_gpu = cuda.mem_alloc(column_sums.nbytes)
    normalized_columns_gpu = cuda.mem_alloc(columns.nbytes)

    # Transferimos datos a la GPU
    cuda.memcpy_htod(columns_gpu, columns)
    cuda.memcpy_htod(column_sums_gpu, column_sums)

    # Definimos el tamaño del bloque y la cuadrícula
    block_size = (32, 32, 1)
    grid_size = (int(np.ceil(cols / block_size[0])), int(np.ceil(rows / block_size[1])), 1)

    # Llamamos al kernel
    normalize_columns_kernel(columns_gpu, column_sums_gpu, normalized_columns_gpu, np.int32(rows), np.int32(cols), block=block_size, grid=grid_size)

    # Creamos un array para almacenar el resultado
    normalized_columns = np.empty_like(columns)

    # Transferimos el resultado de vuelta a la CPU
    cuda.memcpy_dtoh(normalized_columns, normalized_columns_gpu)

    return normalized_columns

# Ejemplo de uso
if __name__ == "__main__":
    exponent = 2.0
    x  = np.random.rand(5, 3)
    dos= normalize_power_columns(x, exponent)
      # Matriz de ejemplo
    print("Original columns:")
    print(x)

    #normalized_columns = normalize_columns_gpu(x)
    normalized_columns2 = x/np.sum(x, axis=0, keepdims=1)
    print(normalized_columns2)
    print("Normalized columns:")
    print(dos)
