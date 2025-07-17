import numpy as np
import cv2


image = cv2.imread("img/barco.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(type(image))