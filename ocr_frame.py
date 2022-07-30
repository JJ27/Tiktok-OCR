import cv2
from pytesseract import image_to_string
import numpy as np


def getText(img):
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(HSV_img)
    v = cv2.GaussianBlur(v, (1, 1), 0)
    thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imwrite('{}.png'.format(filename), thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 2))
    thresh = cv2.dilate(thresh, kernel)
    txt = image_to_string(thresh, config="--psm 6")
    return txt