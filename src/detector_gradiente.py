from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
 
rng.seed(12345)
 
def dibujar_cuadrado(contours,image):
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
 
def thresh_callback(val,image):
    threshold = val

    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    cv.imshow('canny', canny_output)

    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dibujar_cuadrado(contours,image)


    # Show in a window
    cv.imshow('Contours', image)
 
# Load source image
# parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
# args = parser.parse_args()
 
src = cv.imread("/home/tille/Desktop/Tesis/code/img/barco.jpg")
src = cv.resize(src,(640,480))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
 
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
 
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 170 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh,src)
 
cv.waitKey()