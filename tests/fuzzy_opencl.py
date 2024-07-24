import numpy as np
import pyopencl as cl

# Definir el código del kernel OpenCL
kernel_code = """
__kernel void fuzzy_edge_detect(__global float* fuzzy_image,
                                __global float* edge_image,
                                __global int* neighbors,
                                const int rows,
                                const int cols)
{
    int i = get_global_id(0);  // Índice global en la dimensión x
    int j = get_global_id(1);  // Índice global en la dimensión y

    if (i > 0 && i < rows-1 && j > 0 && j < cols-1)
    {
        // Calcular el índice para fuzzy_image y edge_image
        int idx = i * cols + j;

        // Calcular los valores de los vecinos
        float neighbor_values[9];
        for (int k = 0; k < 9; ++k) {
            int di = neighbors[k*2];
            int dj = neighbors[k*2 + 1];
            neighbor_values[k] = fuzzy_image[(i+di)*cols + (j+dj)];
        }

        // Aplicar lógica difusa (aquí deberías usar skfuzzy o tu lógica personalizada)
        edge_image[idx] = neighbor_values[0];  // Ejemplo de asignación directa
    }
}
"""

# Definir las dimensiones de la imagen
rows, cols = 300, 300

# Crear contexto y cola de comandos para OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Compilar el kernel OpenCL
program = cl.Program(context, kernel_code).build()

# Preparar datos en el host (CPU)
fuzzy_image = np.random.rand(rows, cols).astype(np.float32)
edge_image = np.zeros_like(fuzzy_image).astype(np.float32)
neighbors = np.array([-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1], dtype=np.int32)

# Crear buffers en la GPU para los datos
fuzzy_image_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fuzzy_image)
edge_image_gpu = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, edge_image.nbytes)
neighbors_gpu = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=neighbors)

# Ejecutar el kernel OpenCL
program.fuzzy_edge_detect(queue, fuzzy_image.shape, None,
                          fuzzy_image_gpu, edge_image_gpu, neighbors_gpu, np.int32(rows), np.int32(cols))

# Transferir resultados de la GPU al host
cl.enqueue_copy(queue, edge_image, edge_image_gpu).wait()

# Mostrar o trabajar con edge_image (resultado)
print(edge_image)