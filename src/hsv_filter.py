import skfuzzy as fuzz
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, abspath, join
from sys import argv

def filter_h(img):
    #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h,_,_ = cv2.split(img_hsv)
    
    h = 255 - h

    hist, bins = np.histogram(h.ravel(), 256, [0, 256])

    max_index = np.argmax(hist)
    most_frequent_intensity = bins[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    h_filtrada = np.copy(h)
    h_filtrada[(h >= most_frequent_intensity-10) & (h <= most_frequent_intensity+10)] = 0

    return h_filtrada

def filter_s(img_path):
    #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)
    
    _,s,_ = cv2.split(img_hsv)
    
    s = 255 - s

    hist, bins = np.histogram(s.ravel(), 256, [0, 256])

    max_index = np.argmax(hist)
    most_frequent_intensity = bins[max_index]

    # Se asigna un rango de 20 pixeles entorno a este
    s_filtrada = np.copy(s)
    s_filtrada[(s >= most_frequent_intensity-10) & (s <= most_frequent_intensity+10)] = 0

    return s_filtrada
if __name__ == "__main__":

    # Ruta a la imagen
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/amanecer.jpeg")
    img=cv2.imread(filename)
    
    cv2.imshow('imagen final.png',filter_s(img))
    cv2.waitKey(0)
