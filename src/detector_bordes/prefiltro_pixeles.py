import cupy as cp
import cv2

def normalize_image(image:cp.array) -> cp.array:
    """
    Normalize the pixel values of an image to the range [0, 1].
    This function takes an image represented as a CuPy array and normalizes its pixel values 
    such that the minimum value becomes 0 and the maximum value becomes 1.
    Args:
        image (cp.array): The input image to be normalized.
    Returns:
        cp.array: The normalized image with pixel values in the range [0, 1].
    """
    
    min_val = cp.min(image)
    max_val = cp.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)

    return normalized_image

def filter_image(image,kernel_code):
    # Cargar la imagen y convertirla a escala de grises
    input_image = cv2.resize(image, (256, 256))
    height, width = input_image.shape

    # Alocar memoria en la GPU
    d_input = cp.array(input_image)
    d_output = cp.empty((256, 256), dtype=cp.uint8)

    # Definir el tamaño del bloque y el tamaño de la rejilla
    block_size = 256
    grid_size = (256 * 256 + block_size - 1) // block_size

    # Lanzar el kernel
    process_image_kernel = cp.RawKernel(kernel_code, 'process_image')
    process_image_kernel((grid_size,), (block_size,), (d_input, d_output, width, height))

    # Obtener el resultado de vuelta al host
    output_image = cp.asnumpy(d_output)

    return output_image

