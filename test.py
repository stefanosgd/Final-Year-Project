from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())


def mouseRGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorsB = frame[y, x, 0]
        colorsG = frame[y, x, 1]
        colorsR = frame[y, x, 2]
        colors = frame[y, x]
        print("Red: ", colorsR)
        print("Green: ", colorsG)
        print("Blue: ", colorsB)
        print("BRG Format: ", colors)
        print("Coordinates of pixel: X: ", x, "Y: ", y)


cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB', mouseRGB)

capture = cv2.VideoCapture(args["video"])
# capture = cv2.VideoCapture(0)

while (True):

    ret, frame = capture.read()

    cv2.imshow('mouseRGB', frame)

    if cv2.waitKey(0) == 27:
        break

capture.release()
cv2.destroyAllWindows()
