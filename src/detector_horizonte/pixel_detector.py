import cupy as cp
import numpy as np
import time

from collections import Counter
from PIL import Image

from utils import read_image


def fuzzy_difference(a, b):
    return np.sum(np.abs(a - b))

def fuzzy_difference_cp(a,b):
    return cp.sum(cp.abs(a-b))

def find_largest_fuzzy_jump_orig(pixels_list):
    pixels_list = pixels_list.tolist()
    max_jump = 0
    max_jump_index = 0
    for i in range(1, len(pixels_list)):
        jump = fuzzy_difference(np.array(pixels_list[i]), np.array(pixels_list[i-1]))
        if jump > max_jump:
            max_jump = jump
            max_jump_index = i + 76
    return max_jump_index

def find_largest_fuzzy_jump(pixels:np.array):
    pixels = pixels.astype(np.int16)
    jumps = np.array([fuzzy_difference(pixels[i], pixels[i - 1])
                      for i in range(1, len(pixels))])

    # Encontrar el índice del salto más grande
    max_jump_index = np.argmax(jumps) + 1 + 76

    return max_jump_index

def find_largest_fuzzy_jump_cp(pixels:cp.array):
    jumps = fuzzy_difference(pixels[1:], pixels[:-1])

    # Encontrar el índice del mayor salto
    max_jump_index = cp.argmax(jumps) + 1 + 76

    # Convertir el índice a int para asegurar compatibilidad con funciones que esperan enteros normales
    return int(max_jump_index)


def get_pixels(image:np.array):
    # 0.097 seg
    if len(image.shape) == 3:
        flattened_image = image.reshape(-1, 3)
    else:
        flattened_image = image.ravel()

    # Utilizar numpy.unique para contar las ocurrencias de cada píxel
    unique_pixels, counts = np.unique(flattened_image, axis=0, return_counts=True)

    # Crear una lista de tuplas (pixel, cantidad) y ordenar por cantidad
    sorted_pixel_counts = sorted(zip(map(tuple, unique_pixels), counts), key=lambda x: x[1], reverse=True)

    return sorted_pixel_counts

def get_pixels_pil(image: Image.Image):
    # 0.035 seg WIN
    # Utilizar directamente getdata() sin convertir a lista
    pixel_counts = Counter(image.getdata())

    # Ordenar los píxeles por ocurrencia en orden descendente
    sorted_pixel_counts = sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_pixel_counts

def get_pixels_cp(image: cp.ndarray):
    # Aplanar la imagen en un array bidimensional de forma (n, 3)
    flattened_image = image.reshape(-1, 3)
    
    # Convertir cada píxel (R, G, B) en un número único utilizando un enfoque basado en potencias de 256
    unique_values = (flattened_image[:, 0] << 16) + (flattened_image[:, 1] << 8) + flattened_image[:, 2]

    # Contar las ocurrencias de cada valor único utilizando cupy.bincount
    counts = cp.bincount(unique_values)

    # Obtener los índices de los valores únicos y sus cuentas
    unique_pixels = cp.nonzero(counts)[0]
    pixel_counts = counts[unique_pixels]

    # Convertir a un diccionario manteniendo el procesamiento en la GPU
    keys = unique_pixels.get()
    values = pixel_counts.get()
    pixel_dict = dict(zip(keys, values))

    return pixel_dict

def separate_pixels(image:np.array):
    # 7.03*10-5 seg WIN

    # Extraer los píxeles de la izquierda, derecha y centro
    left_pixels = image[76:116, 0, :]
    right_pixels = image[76:116, -1, :]
    center_pixels = image[76:116, 256 // 2, :]

    return left_pixels,center_pixels,right_pixels

def separate_pixels_cp(image:cp.array):
    # 0.719 seg
    left_pixels = image[76:116, 0, :]
    right_pixels = image[76:116, -1, :]
    center_pixels = image[76:116, 256 // 2, :]

    return left_pixels,center_pixels,right_pixels

###### MAIN ###########
def main():
    image_path = "/home/tesis_liz_tille/detector-obstaculos-logica-difusa/assets/images/akaso2.jpeg"
    image:cp.array = read_image(image_path,256,192)

def time_test():
    start = time.time()
    image_path = "/home/tesis_liz_tille/detector-obstaculos-logica-difusa/assets/images/akaso2.jpeg"
    image:cp.array = read_image(image_path,256,192)
    
    image_pil = Image.open(image_path)
    # Resize the image to 256x192 pixels
    image_pil = image_pil.resize((256, 192))
    image_np = cp.asnumpy(image)

    ########### get pixels #############
    start_pixel_cp = time.time()
    pixel_dict=get_pixels_cp(image)
    end_pixel_cp = time.time()

    start_pixel = time.time()
    pixel_dict=get_pixels(image_np)
    end_pixel = time.time()

    start_pixel_pil = time.time()
    pixel_dict=get_pixels_pil(image_pil)
    end_pixel_pil = time.time()
    #####################################
    
    ######### separate pixels and largest jump ############
    start_sep_np = time.time()
    left,center,right = separate_pixels(image_np)
    end_sep_np = time.time()

    left_max_fuzzy_jump_index = find_largest_fuzzy_jump(left)
    center_max_fuzzy_jump_index = find_largest_fuzzy_jump(center)
    right_max_fuzzy_jump_index = find_largest_fuzzy_jump(right)

    end_lar_np = time.time()

    start_sep_orig = time.time()
    left,center,right = separate_pixels(image)
    end_sep_orig = time.time()

    left_max_fuzzy_jump_index = find_largest_fuzzy_jump_orig(left)
    center_max_fuzzy_jump_index = find_largest_fuzzy_jump_orig(center)
    right_max_fuzzy_jump_index = find_largest_fuzzy_jump_orig(right)

    end_lar_orig = time.time()

    start_sep_cp = time.time()
    left,center,right = separate_pixels_cp(image)
    end_sep_cp = time.time()

    left_max_fuzzy_jump_index = find_largest_fuzzy_jump_cp(left)
    center_max_fuzzy_jump_index = find_largest_fuzzy_jump_cp(center)
    right_max_fuzzy_jump_index = find_largest_fuzzy_jump_cp(right)
    end_lar_cp = time.time()
    ######################################

    end = time.time()
    print(f"separate_pixels_np: {end_sep_np - start_sep_np} segundos.")
    print(f"separate_pixels_cp: {end_sep_cp - start_sep_cp} segundos.")
    print(f"largest_jump_np: {end_lar_np - start_sep_np} segundos.")
    print(f"largest_jump_orig: {end_lar_orig - start_sep_orig} segundos.")
    print(f"largest_jump_cp: {end_lar_cp - start_sep_cp} segundos.")
    print("--------------")
    print(f"get_pixels_pil: {end_pixel_pil - start_pixel_pil} segundos.")
    print(f"get_pixels_cp: {end_pixel_cp - start_pixel_cp} segundos.")
    print(f"get_pixels: {end_pixel - start_pixel} segundos.")
    print(f"Tardó en total: {end - start} segundos.")

if __name__ == "__main__":
    time_test()