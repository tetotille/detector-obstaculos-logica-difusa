extern "C" __global__
void triangular_low(const unsigned char* input, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int limite = 128;

    if (idx < 256 * 256) {
        // Aplicar la logica de procesamiento
        unsigned int pixel_value = input[idx];
        output[idx] = (pixel_value <= limite) ? static_cast<float> (1 - (pixel_value / limite)) : 0.0f;
    }
}