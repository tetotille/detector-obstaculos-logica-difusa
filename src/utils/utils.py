import numpy as np
import cv2
from scipy.ndimage import label

try:
    import cupy as cp
except:
    import numpy as cp
    print("cuda no está instalado.")


# hacer_mascara_kernel = cp.RawKernel(open("kernels/hacer_mascara_kernel.cu").read(), "hacer_mascara_kernel")

class FuzzyImage:
    def __init__(self,img_path:str):
        self.image = cv2.imread(img_path)
        if self.image is None:
            raise ValueError("No se pudo cargar la imagen. Verifique la ruta del archivo y asegúrese de que el archivo exista.")
        self.image = self.__resize_image(256)
    
    def __resize_image(self,new_width:int):
        orig_height, orig_width, channels = self.image.shape
        new_height = int(orig_height * new_width / orig_width)
        return cv2.resize(self.image, (new_width, new_height))


def crop_horizontal(imagen, indice_vertical):
    """
    Recorta una imagen a color horizontalmente en un índice dado.
    Args:
        imagen: Una imagen a color en formato NumPy.
        indice_vertical: El índice vertical donde se realizará el recorte.
    Returns:
        Una tupla que contiene dos imágenes: la parte superior y la parte inferior.
    """

    if indice_vertical < 0 or indice_vertical >= imagen.shape[0]:
        raise ValueError("El índice vertical está fuera de los límites de la imagen.")
    parte_superior = imagen[:indice_vertical, :]
    parte_inferior = imagen[indice_vertical:, :]
    return parte_superior, parte_inferior

def hacer_mascara(image3, mask):
    height, width, channels = image3.shape
    mask_height, mask_width = mask.shape

    # Crear imágenes para almacenar los resultados
    contour_image = cp.zeros((height, width, channels), dtype=cp.uint8)
    rgba_image = cp.zeros((height, width, 4), dtype=cp.uint8)

    # Configuración de bloques e hilos para el kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Ejecutar el kernel
    # hacer_mascara_kernel(
    #     blocks_per_grid, threads_per_block,
    #     (image3, mask, contour_image, rgba_image, width, height, channels, mask_width, mask_height)
    # )

    return contour_image, rgba_image

def read_image(image:np.array,new_width:int,new_height:int,**params) -> tuple[np.array,cp.array]:
    """
    Lee una imagen de un archivo y la convierte en un array de CuPy.
    Args:
        image_path: La ruta del archivo de imagen.
    Returns:
        La imagen como una tupla de arrays de NumPy y CuPy.
    """
    normalize = params.get("normalize", False)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifique la ruta del archivo y asegúrese de que el archivo exista.")
    
    # Redimensionar la imagen al nuevo ancho
    if new_height is None:
        new_height = int(image.shape[0] * new_width / image.shape[1])
    image = cv2.resize(image, (new_width, new_height))
    if normalize:
        image = image / 255.0
    return image,cp.asarray(image)

def resize_image_bgr(image, new_shape):
    orig_height, orig_width, channels = image.shape
    new_height, new_width = new_shape

    scale_y = orig_height / new_height
    scale_x = orig_width / new_width

    y = cp.arange(new_height) * scale_y
    x = cp.arange(new_width) * scale_x
    x_grid, y_grid = cp.meshgrid(x, y)

    x0 = cp.floor(x_grid).astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, orig_width - 1)
    y0 = cp.floor(y_grid).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, orig_height - 1)

    x_weight = x_grid - x0
    y_weight = y_grid - y0

    resized_image = cp.zeros((new_height, new_width, channels), dtype=image.dtype)
    for c in range(channels):
        Ia = image[y0, x0, c]
        Ib = image[y1, x0, c]
        Ic = image[y0, x1, c]
        Id = image[y1, x1, c]

        resized_image[:, :, c] = (
            Ia * (1 - x_weight) * (1 - y_weight) +
            Ib * (1 - x_weight) * y_weight +
            Ic * x_weight * (1 - y_weight) +
            Id * x_weight * y_weight
        )

    resized_image = cp.clip(resized_image, 0, 255)
    return resized_image.astype(cp.uint8)

def block_framed(mask_max_cluster_cpu):
    x_block = 20
    y_block = 6
    mask_matrix = cp.zeros((y_block,x_block), dtype=cp.uint8)
    height, width = mask_max_cluster_cpu.shape
    cuadros = []
    for y in range(y_block):
        for x in range(x_block):
            mask_matrix[y,x] = cp.count_nonzero(mask_max_cluster_cpu[y*(height//y_block):(y+1)*(height//y_block),x*(width//x_block):(x+1)*(width//x_block)])
            if mask_matrix[y,x] > 20:
                tiene_vecino = False
                for cuadro in cuadros:
                    if (y,x) in cuadro["vecinos"]:
                        nuevos_vecinos = {(y-1,x),(y+1,x),(y,x-1),(y,x+1)}
                        cuadro["vecinos"] = cuadro["vecinos"].union(nuevos_vecinos)
                        cuadro["x_init"] = min(cuadro["x_init"],x*(width//x_block))
                        cuadro["x_end"] = max(cuadro["x_end"],(x+1)*(width//x_block))
                        cuadro["y_init"] = min(cuadro["y_init"],y*(height//y_block))
                        cuadro["y_end"] = max(cuadro["y_end"],(y+1)*(height//y_block))
                        cuadro["centroid"] = (cuadro["x_init"]+cuadro["x_end"])//2,(cuadro["y_init"]+cuadro["y_end"])//2
                        cuadro["weight"] = cuadro["weight"] + mask_matrix[y,x]

                        tiene_vecino = True
                if not tiene_vecino:
                    cuadros.append({"vecinos":{(y-1,x),(y+1,x),(y,x-1),(y,x+1)},"x_init":x*(width//x_block),"x_end":(x+1)*(width//x_block),"y_init":y*(height//y_block),"y_end":(y+1)*(height//y_block),
                                    "centroid":((x*(width//x_block)+(x+1)*(width//x_block))//2,(y*(height//y_block)+(y+1)*(height//y_block))//2),"weight":mask_matrix[y,x]})
    return cuadros

def neighbor_framed_np(mask_max_cluster_cpu):
    # Identificar regiones conectadas usando etiquetado
    structure = cp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # Conexión 8
    labeled, num_features = label(mask_max_cluster_cpu != 0, structure=structure)
    
    cuadros = []
    for label_id in range(1, num_features + 1):
        # Extraer región con la etiqueta actual
        region_mask = (labeled == label_id)
        indices = cp.argwhere(region_mask)  # Obtener índices de los píxeles no cero
        
        if indices.shape[0] >= 30:  # Filtrar regiones pequeñas
            y_coords, x_coords = indices[:, 0], indices[:, 1]
            x_init, x_end = cp.min(x_coords), cp.max(x_coords)
            y_init, y_end = cp.min(y_coords), cp.max(y_coords)
            weight = indices.shape[0]
            x_centroid = cp.mean(x_coords)
            y_centroid = cp.mean(y_coords)
            
            cuadro = {
                "puntos": None,  # Si necesitas puntos específicos, usar `indices.tolist()` (costoso en memoria)
                "x_init": x_init,
                "x_end": x_end,
                "y_init": y_init,
                "y_end": y_end,
                "x": cp.sum(x_coords),
                "y": cp.sum(y_coords),
                "weight": weight,
                "x_centroid": int(x_centroid),
                "y_centroid": int(y_centroid),
            }
            cuadros.append(cuadro)

    return cuadros

def neighbor_framed(mask_max_cluster_cpu):
    non_zero_positions = set(zip(*cp.nonzero(mask_max_cluster_cpu)))
    cuadros = []
    while non_zero_positions:
        pixel = non_zero_positions.pop()
        cuadro = {"puntos":set(),"x_init":pixel[1],"x_end":pixel[1],"y_init":pixel[0],"y_end":pixel[0],"x":pixel[1],"y":pixel[0],"weight":1}
        vecinos = {(pixel[0]-1,pixel[1]),(pixel[0]+1,pixel[1]),(pixel[0],pixel[1]-1),(pixel[0],pixel[1]+1),(pixel[0]-1,pixel[1]-1),(pixel[0]-1,pixel[1]+1),(pixel[0]+1,pixel[1]-1),(pixel[0]+1,pixel[1]+1)}
        while vecinos:
            vecino = vecinos.pop()
            if vecino[0] < 0 or vecino[1] < 0 or vecino[0] >= mask_max_cluster_cpu.shape[0] or vecino[1] >= mask_max_cluster_cpu.shape[1]: continue
            if mask_max_cluster_cpu[vecino[0],vecino[1]] == 255:
                cuadro["puntos"].add(vecino)
                cuadro["weight"] += 1
                cuadro["x_init"] = min(cuadro["x_init"],vecino[1])
                cuadro["x_end"] = max(cuadro["x_end"],vecino[1])
                cuadro["y_init"] = min(cuadro["y_init"],vecino[0])
                cuadro["y_end"] = max(cuadro["y_end"],vecino[0])
                cuadro["x"] += vecino[1]
                cuadro["y"] += vecino[0]
                non_zero_positions.discard(vecino)
                vecinos = vecinos.union({(vecino[0]-1,vecino[1]),(vecino[0]+1,vecino[1]),(vecino[0],vecino[1]-1),(vecino[0],vecino[1]+1),(vecino[0]-1,vecino[1]-1),(vecino[0]-1,vecino[1]+1),(vecino[0]+1,vecino[1]-1),(vecino[0]+1,vecino[1]+1)}) - cuadro["puntos"]
        cuadro["x_centroid"] = cuadro["x"]//cuadro["weight"]
        cuadro["y_centroid"] = cuadro["y"]//cuadro["weight"]
        cuadros.append(cuadro)
        cuadros = [cuadro for cuadro in cuadros if cuadro["weight"] > 100]
    return cuadros

if __name__ == "__main__":
    # Leer la imagen y convertirla a un array de CuPy
    image_path = "/home/tesis_liz_tille/detector-obstaculos-logica-difusa/assets/images/akaso1.jpeg"
    image = read_image(image_path, 256)
    cv2.imwrite("result.jpg", cp.asnumpy(image))