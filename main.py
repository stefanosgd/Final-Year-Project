# import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time
from pprint import pprint
from matplotlib import pyplot as plt

pause_playback = False  # pause until key press after each image
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())

# define range of blue color in HSV
# lower_blue = np.array([90, 100, 10])
# upper_blue = np.array([120, 255, 130])
upper_blue = np.array([60, 125, 175])
lower_blue = np.array([0, 105, 140])

# define range of yellow color in HSV
# lower_yellow = np.array([20, 200, 150])
# upper_yellow = np.array([30, 255, 230])
upper_yellow = np.array([180, 175, 70])
lower_yellow = np.array([120, 130, 30])

# define range of green color in HSV
# lower_green = np.array([50, 100, 10])
# upper_green = np.array([115, 155, 50])
upper_green = np.array([255, 129, 129])
lower_green = np.array([100, 113, 113])

# define range of red color in HSV
# lower_red = np.array([160, 200, 20])
# upper_red = np.array([190, 255, 60])
upper_red = np.array([255, 131, 176])
lower_red = np.array([100, 111, 153])

weight_diameter = 450 / 1000  # m
start_x = 0
start_y = 0

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
# pts = deque(maxlen=args["buffer"])
pts = []
# counter = 0
# (dX, dY) = (0, 0)
# direction = ""
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = cv2.VideoCapture(0)
    videoStream = True

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
    videoStream = False

# create the overlay path
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
# img = np.zeros((height, width, 3), np.uint8)
# cv2.namedWindow("Path")

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    if videoStream:
        ret, frame = vs.read()
    else:
        ret, frame = vs.read()
        frame = frame[0:frame.shape[0], 300:(frame.shape[1] - 300)]
        # print(frame.shape)
        frame = ret, frame

    cv2.namedWindow('Frame')

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # todo Contours here
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # Threshold the HSV image to get only blue colors
    # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # # Threshold the HSV image to get only yellow colors
    # yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # # Threshold the HSV image to get only green colors
    # green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # # Threshold the HSV image to get only red colors
    # red_mask = cv2.inRange(hsv, lower_red, upper_red)

    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(ycbcr, lower_blue, upper_blue)
    # Threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(ycbcr, lower_yellow, upper_yellow)
    # Threshold the HSV image to get only green colors
    # green_mask = cv2.inRange(ycbcr, lower_green, upper_green)
    # Threshold the HSV image to get only red colors
    # red_mask = cv2.inRange(ycbcr, lower_red, upper_red)

    mask = blue_mask + yellow_mask
    # mask = blue_mask + yellow_mask + green_mask + red_mask
    # cv2.imshow("Original mask", mask)

    mask = cv2.erode(mask, None, iterations=3)

    mask = cv2.dilate(mask, None, iterations=3)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only processed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(x), int(y))
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # cv2.circle(img, center, 5, (0, 0, 255), -1)

        # print(radius)
        # only proceed if the radius meets a minimum size
        if 50 < radius < 90:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            # pts.appendleft(center)
            pts.insert(0, center)
            # loop over the set of tracked points
            if len(pts) == 1:
                displacement_x = []
                displacement_y = []
                velocity_x = [0, 0]
                acceleration_x = []
                velocity_y = [0, 0]
                acceleration_y = []
                time_x = []
                time_y = []
                diameter = radius * 2
                start_y, start_x = pts[0]
            else:
                displacement_x_calculated = ((start_x - pts[0][1]) * weight_diameter) / diameter
                displacement_y_calculated = ((pts[0][0] - start_y) * weight_diameter) / diameter
                displacement_x.append(displacement_x_calculated)
                if len(pts) >= 4:
                    change = pts[3][1] - pts[0][1]
                    velocity_x.append(change)
                    change = pts[3][0] - pts[0][0]
                    velocity_y.append(change)
                # if len(velocity_x) >= 2:
                change = velocity_x[-1] - velocity_x[-2]
                acceleration_x.append(change)
                change = velocity_y[-1] - velocity_y[-2]
                acceleration_y.append(change)
                time_x.append(len(pts) / 25)
                displacement_y.append(displacement_y_calculated)
                time_y.append(len(pts) / 25)

        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

    # todo HoughCircles here

    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1.2, 1000, minRadius=40, maxRadius=90)
    #     # ensure at least some circles were found
    #     if circles is not None:
    #         # convert the (x, y) coordinates and radius of the circles to integers
    #         circles = np.round(circles[0, :]).astype("int")
    #
    #         # loop over the (x, y) coordinates and radius of the circles
    #         for (x, y, r) in circles:
    #             # draw the circle in the output image, then draw a rectangle
    #             # corresponding to the center of the circle
    #             cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    #             cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Res", res)
    # cv2.imshow("Path", img)
    key = cv2.waitKey(15 * (not pause_playback)) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    elif key == ord("r"):
        pts = []
    elif key == ord(" "):
        # pprint(displacement_x)
        # print(pts[-1], pts[0])
        pause_playback = not pause_playback

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
plt.plot(displacement_y, displacement_x)
plt.axes().set_aspect('equal')
# pprint(displacement_x)
# plt.subplot(321)
# plt.plot(time_x, displacement_x)
# plt.ylabel("Displacement")
# plt.xlabel("Time")
# plt.subplot(322)
# plt.plot(time_y, displacement_y)
# plt.ylabel("Displacement")
# plt.xlabel("Time")
# plt.subplot(323)
# plt.plot(time_x, velocity_x)
# plt.subplot(324)
# plt.plot(time_y, velocity_y)
# plt.subplot(325)
# plt.plot(time_x, acceleration_x)
# plt.subplot(326)
# plt.plot(time_y, acceleration_y)
# plt.plot(time_x, np.divide(displacement_x, time_x))
# plt.subplot(224)
# plt.plot(time_y, np.divide(displacement_y, time_y))
plt.show()
