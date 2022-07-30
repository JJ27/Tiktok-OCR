from ocr_frame import getText
from imutils.video import VideoStream
from imutils.perspective import four_point_transform
from pytesseract import image_to_string
import pytesseract
import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-o", "--output", help="path to the output file")
ap.add_argument("-c", "--min_conf", type=int, default=50, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

outputBuilder = None
writer = None
outputW = None
outputH = None

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
time.sleep(1.0)

#loop video frames
while True:
    orig = vs.read()
    orig = orig[1]
    if orig is None:
        break
    frame = imutils.resize(orig, width=600)
    prefinal = getText(frame)
    if "\n" in prefinal:
        final = prefinal.split("\n")
        for i, item in enumerate(prefinal):
            if("@" in item):
                print(prefinal[i+1]);
    else:
        continue

