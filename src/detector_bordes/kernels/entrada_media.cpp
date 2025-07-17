extern "C" __global__
void process_image(const unsigned char* input, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 256 * 256) {
        // Aplicar la logica de procesamiento
        int sigma = 42;
        int miu = 128;

        double exponent = -((input[idx] - miu) * (input[idx] - miu)) / (2 * sigma * sigma);

        output[idx] = exp(exponent);
    }
}