import cupy as cp
import cv2
from os.path import dirname, abspath, join
from sys import argv
import detectar_horizonte2
import utils
import cupy as cp

def bgr_to_hsv(img):
    """
    Convierte una imagen BGR a HSV.
    img: array 3D de CuPy con forma (height, width, 3) y valores en el rango [0, 255].
    """
    img = img / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # Cambiar el orden de los canales
    
    max_val = cp.max(img, axis=2)
    min_val = cp.min(img, axis=2)
    delta = max_val - min_val

    h = cp.zeros_like(max_val)

    h = cp.where(max_val == r, 60.0 * (((g - b) / (delta + 1e-6)) % 6), h)
    h = cp.where(max_val == g, 60.0 * (((b - r) / (delta + 1e-6)) + 2), h)
    h = cp.where(max_val == b, 60.0 * (((r - g) / (delta + 1e-6)) + 4), h)

    h = cp.where(delta == 0, 0.0, h)
    h = h / 2.0  # Convertir de [0, 360] a [0, 180] como en OpenCV

    s = cp.where(max_val == 0, 0.0, delta / max_val)
    v = max_val

    h = cp.asarray(h * 255 / 180, dtype=cp.uint8)
    s = cp.asarray(s * 255, dtype=cp.uint8)
    v = cp.asarray(v * 255, dtype=cp.uint8)

    return h, s, v

def interp_membership(x_intensities, mf, h):
    """
    Interpola la membresía en una función de membresía dada.
    
    x_intensities: Array de CuPy con los valores en los que se ha evaluado la función de membresía.
    mf: Array de CuPy con los valores de la función de membresía.
    h: Array de CuPy con los valores a interpolar.
    
    Retorna un array de CuPy con los grados de pertenencia interpolados.
    """
    # Asegúrate de que h está dentro de los límites de x_intensities
    h = cp.clip(h, x_intensities[0], x_intensities[-1])
    
    # Encontrar los índices de los vecinos más cercanos en x_intensities
    idx = cp.searchsorted(x_intensities, h) - 1
    idx = cp.clip(idx, 0, len(x_intensities) - 2)
    
    x0 = x_intensities[idx]
    x1 = x_intensities[idx + 1]
    mf0 = mf[idx]
    mf1 = mf[idx + 1]
    
    # Interpolación lineal
    membership_values = mf0 + (mf1 - mf0) * (h - x0) / (x1 - x0 + 1e-6)
    
    return membership_values

def trapmf(x, a, b, c, d):
    """
    Calcula la función de membresía trapezoidal.
    x: array de CuPy
    a, b, c, d: puntos de la función trapezoidal
    """
    y = cp.zeros_like(x)
    y = cp.where((x >= a) & (x < b), (x - a) / (b - a + 1e-6), y)
    y = cp.where((x >= b) & (x <= c), cp.ones_like(x), y)
    y = cp.where((x > c) & (x <= d), (d - x) / (d - c + 1e-6), y)
    return y

def filter_h(img):

    h, _, _ = bgr_to_hsv(img)
    
    h = cp.asarray(255 - h)

    num_bins = 256
    x_intensities = cp.arange(0, num_bins, 1)

    # Crear los límites de los trapezoides para todos los bins
    a = cp.concatenate((cp.array([0]), x_intensities[:-1]))[:, cp.newaxis]
    b = x_intensities[:, cp.newaxis]
    c = cp.clip(x_intensities + 1, 0, num_bins - 1)[:, cp.newaxis]
    d = cp.concatenate((x_intensities[1:], cp.array([num_bins - 1])))[:, cp.newaxis]

    # Expansión de h para hacer broadcast
    h_expanded = h.ravel()[cp.newaxis, :]
    
    # Cálculo de la función de membresía trapezoidal en una sola operación vectorizada
    membership_values = trapmf(h_expanded, a, b, c, d)

    # Sumar los valores de membresía para formar el histograma difuso
    fuzzy_hist = cp.sum(membership_values, axis=1)
    max_index = cp.argmax(fuzzy_hist)
    most_frequent_intensity = x_intensities[max_index]

    h_filtrada = cp.copy(h)
    h_filtrada[(h >= most_frequent_intensity-10) & (h <= most_frequent_intensity+10)] = 0
    h=(h_filtrada)
# Suponiendo que `image` es tu imagen en formato RGB cargada como un array de CuPy
# Reshape para convertir la imagen en una lista de colores (cada color es una tupla de 3 valores)
    #print(h.shape)

    fila_interes, imagen = detectar_horizonte2.find_horizontal_line(img)
    _, image3 = utils.crop_horizontal(h, fila_interes)
    return image3, fila_interes

def filter_s(img):

    _, s, _ = bgr_to_hsv(img)
    
    s = cp.asarray(255 - s)

    hist, bins = cp.histogram(s.ravel(), 256, [0, 256])

    max_index = cp.argmax(hist)
    most_frequent_intensity = bins[max_index]

    s_filtrada = cp.copy(s)
    s_filtrada[(s >= most_frequent_intensity-10) & (s <= most_frequent_intensity+10)] = 0
    s = (s_filtrada)
    fila_interes, imagen = detectar_horizonte2.find_horizontal_line(img)
    _, image3 = utils.crop_horizontal(s, fila_interes)

    return image3, fila_interes

# Cargar la imagen usando OpenCV y convertirla a un array de CuPy
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/ypacarai.jpeg")
img = cv2.imread(filename)


# Llamar a las funciones con la imagen
#h_filtrada = filter_h(img)
#s_filtrada = filter_s(img)

# Convertir los resultados de nuevo a imágenes de OpenCV y guardarlas
#h_filtrada_img = cp.asnumpy(h_filtrada).astype('uint8')
#s_filtrada_img = cp.asnumpy(s_filtrada).astype('uint8')

# Guardar las imágenes filtradas con OpenCV
#cv2.imwrite('h_filtrada.png', cv2.cvtColor(h_filtrada_img, cv2.COLOR_RGB2BGR))
#cv2.imwrite('s_filtrada.png', cv2.cvtColor(s_filtrada_img, cv2.COLOR_RGB2BGR))
