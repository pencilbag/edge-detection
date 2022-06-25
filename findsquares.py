import cv2 as cv
import numpy as np
import scipy.signal as ss
import sys
import torch
import math

# Load image, grayscale, median blur, sharpen image
img = cv.imread('C:/Users/Mobno/.spyder-py3/Trythis.png') #stick your image path here

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.medianBlur(gray, 5)
n=-1
sharpen_kernel = np.array([[n,n,n], [n,9,n], [n,n,n]])
#sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv.filter2D(blur, -1, sharpen_kernel)

# Threshold and morph close
thresh = cv.threshold(sharpen, 10, 255, cv.THRESH_BINARY_INV)[1]
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,4))
close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=5)

# Find contours and filter using threshold area
cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 500
max_area = 15000
image_number = 0
for c in cnts:
    area = cv.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        cv.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
        image_number += 1

cv.imshow('sharpen', sharpen)
cv.imshow('close', close)
cv.imshow('thresh', thresh)
cv.imshow('image', img)
cv.waitKey()
