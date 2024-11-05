extern "C" {
    __global__ void cmeans_kernel(float *data, int *clusters, float *membership, int n, int d, int k, float fuzziness) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float *data_ptr = data + i * d;
            float *membership_ptr = membership + i * k;
            float *centroids_ptr = centroids;
            float sum = 0;
            for (int j = 0; j < k; j++) {
                float dist = 0;
                for (int l = 0; l < d; l++) {
                    float diff = data_ptr[l] - centroids_ptr[j * d + l];
                    dist += diff * diff;
                }
                membership_ptr[j] = 1.0f / dist;
                sum += membership_ptr[j];
            }
            for (int j = 0; j < k; j++) {
                membership_ptr[j] = powf(membership_ptr[j] / sum, fuzziness);
            }
        }
    }
}