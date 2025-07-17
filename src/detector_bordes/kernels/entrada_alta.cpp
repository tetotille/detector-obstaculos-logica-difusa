extern "C" __global__
void process_image(const unsigned char* input, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int limite1 = 128;
    int limite2 = 255;

    if (idx < 256 * 256) {
        // Aplicar la logica de procesamiento
        unsigned int pixel_value = input[idx];
        output[idx] = (pixel_value > limite1) ? static_cast<float>(pixel_value - limite1) / (limite2 - limite1) : 0.0f;
    }
}