import torch
import cv2
from os.path import dirname, abspath, join
from sys import argv
import utils_torch

def bgr_to_hsv(img):
    """
    Convierte una imagen BGR a HSV.
    img: tensor 3D de PyTorch con forma (height, width, 3) y valores en el rango [0, 255].
    """
    img = img / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # Cambiar el orden de los canales

    max_val = torch.max(img, dim=2)[0]
    min_val = torch.min(img, dim=2)[0]
    delta = max_val - min_val

    h = torch.zeros_like(max_val)

    # Cálculo del canal H
    h[max_val == r] = 60.0 * (((g[max_val == r] - b[max_val == r]) / (delta[max_val == r] + 1e-6)) % 6)
    h[max_val == g] = 60.0 * (((b[max_val == g] - r[max_val == g]) / (delta[max_val == g] + 1e-6)) + 2)
    h[max_val == b] = 60.0 * (((r[max_val == b] - g[max_val == b]) / (delta[max_val == b] + 1e-6)) + 4)

    h[delta == 0] = 0.0  # Si delta es 0, h debe ser 0
    h = h / 2.0  # Convertir de [0, 360] a [0, 180]

    # Cálculo del canal S
    s = torch.where(max_val == 0, torch.tensor(0.0, dtype=img.dtype), delta / max_val)
    v = max_val

    # Escalar a [0, 255]
    h = (h * 255 / 180).to(torch.uint8)
    s = (s * 255).to(torch.uint8)
    v = (v * 255).to(torch.uint8)

    return h, s, v


def interp_membership(x_intensities, mf, h):
    """
    Interpola la membresía en una función de membresía dada.
    
    x_intensities: tensor de PyTorch con los valores en los que se ha evaluado la función de membresía.
    mf: tensor de PyTorch con los valores de la función de membresía.
    h: tensor de PyTorch con los valores a interpolar.
    
    Retorna un tensor de PyTorch con los grados de pertenencia interpolados.
    """
    # Asegúrate de que h está dentro de los límites de x_intensities
    h = torch.clamp(h, x_intensities[0], x_intensities[-1])
    
    # Encontrar los índices de los vecinos más cercanos en x_intensities
    idx = torch.searchsorted(x_intensities, h) - 1
    idx = torch.clamp(idx, 0, len(x_intensities) - 2)
    
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
    x: tensor de PyTorch
    a, b, c, d: puntos de la función trapezoidal
    """
    y = torch.zeros_like(x)
    
    # Evaluar las condiciones para la función trapezoidal
    y = torch.where((x >= a) & (x < b), (x - a) / (b - a + 1e-6), y)
    y = torch.where((x >= b) & (x <= c), torch.ones_like(x), y)
    y = torch.where((x > c) & (x <= d), (d - x) / (d - c + 1e-6), y)
    
    return y

def filter_s(img, fila_interes):
    """
    Filtra la imagen en el canal de saturación (S) y retorna una imagen filtrada.
    
    img: tensor 3D de PyTorch con la imagen BGR.
    fila_interes: número de fila de interés para el recorte.
    
    Retorna:
    - image3: imagen filtrada.
    - fila_interes: el número de fila de interés.
    """
    _, s, _ = bgr_to_hsv(img)
    
    s = 255 - s.float()  # Asegúrate de convertir a float para la operación
    
    hist = torch.histc(s.view(-1), bins=256, min=0, max=256)  # Crear histograma
    max_index = torch.argmax(hist)
    most_frequent_intensity = max_index.item()

    s_filtrada = s.clone()  # Copiar la saturación
    s_filtrada[(s >= most_frequent_intensity - 10) & (s <= most_frequent_intensity + 10)] = 0
    
    # Convertir a uint8 para la siguiente operación
    s = s_filtrada.to(torch.uint8)

    # Llamar a la función de utils para el recorte
    _, image3 = utils_torch.crop_horizontal(s, fila_interes)

    return image3, fila_interes

# Asegúrate de que la función utils.crop_horizontal esté adaptada para trabajar con tensores de PyTorch.

def filter_h(img, fila_interes):
    """
    Filtra la imagen en el canal de matiz (H) y retorna una imagen filtrada.
    
    img: tensor 3D de PyTorch con la imagen BGR.
    fila_interes: número de fila de interés para el recorte.
    
    Retorna:
    - image3: imagen filtrada.
    - fila_interes: el número de fila de interés.
    """
    h, _, _ = bgr_to_hsv(img)
    
    h = 255 - h.float()  # Asegúrate de convertir a float para la operación

    num_bins = 256
    x_intensities = torch.arange(0, num_bins, 1)

    # Crear los límites de los trapezoides para todos los bins
    a = torch.cat((torch.tensor([0]), x_intensities[:-1]))[:, None]
    b = x_intensities[:, None]
    c = torch.clamp(x_intensities + 1, 0, num_bins - 1)[:, None]
    d = torch.cat((x_intensities[1:], torch.tensor([num_bins - 1])))[:, None]

    # Expansión de h para hacer broadcast
    h_expanded = h.view(-1, 1)
    
    # Cálculo de la función de membresía trapezoidal en una sola operación vectorizada
    membership_values = trapmf(h_expanded, a, b, c, d)

    # Sumar los valores de membresía para formar el histograma difuso
    fuzzy_hist = torch.sum(membership_values, dim=0)
    max_index = torch.argmax(fuzzy_hist)
    most_frequent_intensity = max_index.item()

    h_filtrada = h.clone()
    h_filtrada[(h >= most_frequent_intensity - 10) & (h <= most_frequent_intensity + 10)] = 0
    
    # Suponiendo que `image` es tu imagen en formato RGB cargada como un tensor de PyTorch
    _, image3 = utils_torch.crop_horizontal(h_filtrada.int(), fila_interes)  # Asegúrate de que esté en formato int
    return image3, fila_interes