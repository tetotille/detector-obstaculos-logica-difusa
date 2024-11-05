extern "C" __global__
void hacer_mascara_kernel(
    unsigned char* original, unsigned char* mask, unsigned char* contour_image, unsigned char* rgba_image, 
    int width, int height, int channels, int mask_width, int mask_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        // Crear máscara para los píxeles negros
        bool is_black = original[idx] >= 0 && original[idx] <= 15 &&
                        original[idx + 1] >= 0 && original[idx + 1] <= 15 &&
                        original[idx + 2] >= 0 && original[idx + 2] <= 15;

        // Asignar el valor de blanco o negro según la máscara
        if (is_black) {
            rgba_image[idx] = 255;    // R
            rgba_image[idx + 1] = 255;  // G
            rgba_image[idx + 2] = 255;  // B
        } else {
            rgba_image[idx] = original[idx];
            rgba_image[idx + 1] = original[idx + 1];
            rgba_image[idx + 2] = original[idx + 2];
        }

        // Crear la imagen de contorno en verde (BGR)
        if (x < mask_width && y < mask_height && mask[y * mask_width + x] > 0) {
            contour_image[idx] = 0;       // B
            contour_image[idx + 1] = 255; // G
            contour_image[idx + 2] = 0;   // R
            rgba_image[idx + 3] = 255;    // Alpha
        } else {
            rgba_image[idx + 3] = 0; // Transparencia en el canal alfa
        }
    }
}