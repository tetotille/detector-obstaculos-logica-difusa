import skfuzzy as fuzz
import cv2
import matplotlib.pyplot as plt
import numpy as np


def filter_h(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h,_,_ = cv2.split(img_hsv)
    
    h = 255 - h

    hist, bins = np.histogram(h_filtrada.ravel(), 256, [0, 256])

    max_index = np.argmax(hist)
    most_frequent_intensity = bins[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    h_filtrada = np.copy(h)
    h_filtrada[(h >= most_frequent_intensity-10) & (h <= most_frequent_intensity+10)] = 0

    return h_filtrada

def filter_s(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    _,s,_ = cv2.split(img_hsv)
    
    s = 255 - s

    hist, bins = np.histogram(s_filtrada.ravel(), 256, [0, 256])

    max_index = np.argmax(hist)
    most_frequent_intensity = bins[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    s_filtrada = np.copy(s)
    s_filtrada[(s >= most_frequent_intensity-10) & (s <= most_frequent_intensity+10)] = 0

    return s_filtrada