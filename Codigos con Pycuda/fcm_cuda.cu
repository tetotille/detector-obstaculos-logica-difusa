#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define N 1024  // Número de puntos
#define D 2     // Dimensión de los datos
#define C 4     // Número de clústeres

__device__ float distance(float* a, float* b, int dim) {
    float sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

__global__ void normalize_columns(float* u, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < cols) {
        float sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += u[i * cols + idx];
        }
        for (int i = 0; i < rows; i++) {
            u[i * cols + idx] /= sum;
        }
    }
}

__global__ void compute_membership(float* data, float* centroids, float* membership, int n, int d, int c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float sum = 0.0;
        for (int j = 0; j < c; j++) {
            float dist = distance(data + idx * d, centroids + j * d, d);
            membership[idx * c + j] = dist;
            sum += dist;
        }
        for (int j = 0; j < c; j++) {
            membership[idx * c + j] = 1.0 / (membership[idx * c + j] / sum);
        }
    }
}

__global__ void update_centroids(float* data, float* centroids, float* membership, int n, int d, int c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < c) {
        float numerator[D] = {0};
        float denominator = 0.0;
        for (int i = 0; i < n; i++) {
            float u_ij = membership[i * c + idx];
            for (int j = 0; j < d; j++) {
                numerator[j] += u_ij * data[i * d + j];
            }
            denominator += u_ij;
        }
        for (int j = 0; j < d; j++) {
            centroids[idx * d + j] = numerator[j] / denominator;
        }
    }
}

int main() {
    // Datos de ejemplo
    float h_data[N * D] = { /* ... datos ... */ };
    float h_centroids[C * D] = { /* ... datos ... */ };
    float h_membership[N * C] = {0};

    float *d_data, *d_centroids, *d_membership;
    cudaMalloc(&d_data, N * D * sizeof(float));
    cudaMalloc(&d_centroids, C * D * sizeof(float));
    cudaMalloc(&d_membership, N * C * sizeof(float));

    cudaMemcpy(d_data, h_data, N * D * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    for (int i = 0; i < 100; i++) {
        compute_membership<<<gridSize, blockSize>>>(d_data, d_centroids, d_membership, N, D, C);
        update_centroids<<<gridSize, blockSize>>>(d_data, d_centroids, d_membership, N, D, C);
    }

    cudaMemcpy(h_centroids, d_centroids, C * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_membership);

    return 0;
}
