extern "C" __global__ void membership_kernel(float *input, float *low_output, float *medium_output, float *high_output, int size, float max_pixel, float min_pixel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // Low membership
        if (x <= 0.5) {
            low_output[idx] = 1.0;
        } else {
            low_output[idx] = 0.0;
        }

        // Medium membership
        if (x <= max_pixel / 3) {
            medium_output[idx] = 0.0;
        } else if (x <= (min_pixel + max_pixel) / 2) {
            medium_output[idx] = (x - max_pixel / 3) / ((min_pixel + max_pixel) / 2 - max_pixel / 3);
        } else if (x <= max_pixel * 2 / 3) {
            medium_output[idx] = (max_pixel * 2 / 3 - x) / (max_pixel * 2 / 3 - (min_pixel + max_pixel) / 2);
        } else {
            medium_output[idx] = 0.0;
        }

        // High membership
        if (x >= 0.5) {
            high_output[idx] = 1.0;
        } else {
            high_output[idx] = 0.0;
        }
    }
}