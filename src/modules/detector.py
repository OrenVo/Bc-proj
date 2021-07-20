#!/usr/bin/python3
import sys

import numpy as np
import cv2 as cv
from math import sqrt
from matplotlib import pyplot as plt

MAIN = False
th = None
gl_grey = None
filtered = None
class ObjectsDetector:
    def __init__(self):
        pass


    def detect(self, image):
        # Grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Smoothening image
        diag = sqrt(gray.shape[0] ** 2 + gray.shape[1] ** 2)
        smooth = cv.bilateralFilter(gray, 9, sigmaColor=int(np.mean(np.gradient(gray))), sigmaSpace=int(diag / 50))
        # smooth = cv.medianBlur(gray, 5)

        # Creating mask (black&white image)
        # Histogram based thresholding
        # plt.hist(smooth.ravel(), 256, [0, 256])
        # plt.show()
        # _, mask = cv.threshold(smooth, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)

        # Adaptive thresholding
        # mask = cv.adaptiveThreshold(smooth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 49, 4)
        # Canny-Edge detection
        smooth_gradients = np.gradient(smooth)
        print(np.mean(smooth_gradients), np.max(smooth_gradients), np.min(smooth_gradients),np.median(smooth_gradients), sep=' ')
        mask = cv.Canny(smooth, np.min(smooth_gradients), np.max(smooth_gradients)/2, L2gradient=True)
        # To fill the holes in bone contour
        if MAIN is True:
            global th, gl_grey, filtered
            filtered = cv.cvtColor(smooth, cv.COLOR_GRAY2BGR)
            gl_grey = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        mask = cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 3)
        if MAIN is True:
            th = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        # Contours of object
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # filter smaller contours
        objects = [c for c in contours if cv.contourArea(c) > 2000]
        return objects



if __name__ == '__main__':
    MAIN = True
    # Normal
    #img = cv.imread('/data/public/Bakalářka/dataset/sada1/dorsalis/007_mc_d_1_a.tif')
    # Bad light
    #img = cv.imread('/data/public/Bakalářka/dataset/sada1/radialis/017_mc_s_1_t.tif')
    # Covered
    img = cv.imread('/data/public/Bakalářka/dataset/sada1/radialis/009_mc_s_1_t.tif')
    print(img.shape)
    # Grayscale histogram
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #plt.hist(gray.ravel(), 256, [0, 256])
    #plt.show()


    detector = ObjectsDetector()
    #detector.filter(img)
    #raise BaseException()
    contours = detector.detect(img)
    print(f"Count of contours: {str(len(contours))}")
    #assert len(contours) == 1
    contours.sort(key=cv.contourArea, reverse=True)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, contours, 0, (255, 255, 255), -1000)
    result = cv.bitwise_and(img, img, mask=mask)
    result[mask == 0] = (255, 255, 255)
    cv.drawContours(img, contours, 0, (0, 255, 0), 2)

    concat = np.concatenate((filtered, gl_grey, th, img, result), axis=1)
    cv.imshow("filtered & threshholded & contour & result", concat)
    cv.waitKey(0)

