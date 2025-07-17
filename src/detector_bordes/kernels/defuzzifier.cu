extern "C" __global__
void defuzzify_kernel(const float* fuzzy_matrix, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        int index = idy * width + idx;
        float fuzzy_value = fuzzy_matrix[index];

        // Aplicar una función de defuzzificación, por ejemplo, el método del centroide
        float defuzzified_value = fuzzy_value * 0.5; // Ejemplo simple, ajustar según sea necesario

        output[index] = defuzzified_value;
    }
}