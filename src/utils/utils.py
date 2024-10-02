import numpy as np
import cv2
import cupy as cp
from matplotlib import pyplot as plt

#from fuzzylogic.classes import Domain
#from fuzzylogic.functions import triangular, S, R,bounded_sigmoid

# async def aplicar_logica_difusa_posicion(centroide, dimensiones, mostrar_grafica=False):
#     x, y = centroide
#     ancho, alto = dimensiones

#     pos_x = Domain("x", 0, ancho)
#     pos_y = Domain("y", 0, alto)

#     pos_x.left = S(0, ancho/2)
#     pos_x.center = triangular(ancho/4, (3*ancho)/4)
#     pos_x.right = R(ancho/2,  ancho)

#     pos_y.top = triangular(0, alto/2)
#     pos_y.bottom = triangular(alto/2, alto)

#     grado_izq = pos_x.left(x)
#     grado_centro = pos_x.center(x)
#     grado_der = pos_x.right(x)

#     grado_arriba = pos_y.top(y)
#     grado_abajo = pos_y.bottom(y)

#     pos_x_res = ""
#     pos_y_res = ""

#     if grado_izq >= grado_centro and grado_centro!=0.00:
#         pos_x_res = "centrado hacia la izquierda"
#     elif grado_der >= grado_centro and grado_centro!=0.00:
#         pos_x_res = "centrado hacia la derecha"
#     elif grado_izq > grado_centro and grado_izq > grado_der:
#         pos_x_res = "a la izquierda"
#     elif grado_centro > grado_izq and grado_centro > grado_der:
#         pos_x_res = "en el centro"
#     else:
#         pos_x_res = "a la derecha"

#     if grado_arriba >=  grado_abajo:
#         pos_y_res = "arriba"
#     else:
#         pos_y_res = "abajo"

#     if mostrar_grafica:
#         x_vals = np.arange(0, ancho, 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(x_vals, [pos_x.left(val) for val in x_vals], 'b', linewidth=1.5, label='Izquierda')
#         plt.plot(x_vals, [pos_x.center(val) for val in x_vals], 'g', linewidth=1.5, label='Centro')
#         plt.plot(x_vals, [pos_x.right(val) for val in x_vals], 'r', linewidth=1.5, label='Derecha')
#         plt.title('Funciones de Pertenencia para la Posición Horizontal')
#         plt.xlabel('Coordenada X')
#         plt.ylabel('Grado de Pertenencia')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#         y_vals = np.arange(0, alto, 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(y_vals, [pos_y.top(val) for val in y_vals], 'b', linewidth=1.5, label='Arriba')
#         plt.plot(y_vals, [pos_y.bottom(val) for val in y_vals], 'r', linewidth=1.5, label='Abajo')
#         plt.title('Funciones de Pertenencia para la Posición Vertical')
#         plt.xlabel('Coordenada Y')
#         plt.ylabel('Grado de Pertenencia')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     return f"{pos_y_res} {pos_x_res}", grado_izq, grado_centro, grado_der


# def aplicar_logica_difusa_pixeles(num_pixeles, total_pixeles, mostrar_grafica=False):
#     pocos_umbral = 0.15 * total_pixeles
#     muchos_umbral = 0.10 * total_pixeles

#     pixeles = Domain("pixeles", 0, total_pixeles)

#     pixeles.pocos = bounded_sigmoid(0, muchos_umbral, inverse=True)
#     pixeles.muchos = bounded_sigmoid(pocos_umbral, total_pixeles, inverse=True)

#     grado_pocos = pixeles.pocos(num_pixeles)
#     grado_muchos = pixeles.muchos(num_pixeles)

#     if mostrar_grafica:
#         x = np.arange(0, total_pixeles + 1, 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(x, [pixeles.pocos(val) for val in x], 'b', linewidth=1.5, label='Pocos píxeles')
#         plt.plot(x, [pixeles.muchos(val) for val in x], 'r', linewidth=1.5, label='Muchos píxeles')
#         plt.title('Funciones de Pertenencia para la Cantidad de Píxeles')
#         plt.xlabel('Número de Píxeles')
#         plt.ylabel('Grado de Pertenencia')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     print(f"Grado de pertenencia a pocos píxeles: {grado_pocos}")
#     print(f"Grado de pertenencia a muchos píxeles: {grado_muchos}")

#     return grado_muchos > grado_pocos
        
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

def read_image(image_path:str,new_width:int,**params) -> cp.array:
    """
    Lee una imagen de un archivo y la convierte en un array de CuPy.
    Args:
        image_path: La ruta del archivo de imagen.
    Returns:
        La imagen como un array de CuPy.
    """
    grayscale = params.get("grayscale", False)
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifique la ruta del archivo y asegúrese de que el archivo exista.")
    
    # Redimensionar la imagen al nuevo ancho
    new_height = int(image.shape[0] * new_width / image.shape[1])
    image = cv2.resize(image, (new_width, new_height))
    return cp.asarray(image)