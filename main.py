# import the necessary packages
from collections import deque
import numpy as np
import cv2
import imutils
import time
from matplotlib import pyplot as plt
import winsound
from os import startfile


def track(videoPath, colourMask):

    pause_playback = False  # pause until key press after each image

    upper_black = np.array([100, 135, 155])
    lower_black = np.array([10, 110, 120])

    # define range of blue color in YCrCb
    # lower_blue = np.array([90, 100, 10])
    # upper_blue = np.array([120, 255, 130])
    upper_blue = np.array([60, 130, 150])
    lower_blue = np.array([20, 115, 130])

    # define range of yellow color in YCrCb
    # lower_yellow = np.array([20, 200, 150])
    # upper_yellow = np.array([30, 255, 230])
    upper_yellow = np.array([255, 170, 90])
    lower_yellow = np.array([0, 50, 10])

    # define range of green color in YCrCb
    # lower_green = np.array([50, 100, 10])
    # upper_green = np.array([115, 155, 50])
    upper_green = np.array([255, 110, 130])
    lower_green = np.array([0, 10, 10])

    # define range of red color in YCrCb
    # lower_red = np.array([160, 200, 20])
    # upper_red = np.array([190, 255, 60])
    upper_red = np.array([255, 131, 176])
    lower_red = np.array([100, 111, 153])

    weight_diameter = 450 / 1000  # mm
    start_x = 0
    start_y = 0

    # initialize the list of tracked points, the frame counter,
    # and the coordinate deltas
    pts = []
    # if a video path was not supplied, grab the reference
    # to the webcam
    if videoPath == 0:
        vs = cv2.VideoCapture(0)
        videoStream = True
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(videoPath)
        videoStream = False

    # create the overlay path
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    # allow the camera or video file to warm up
    time.sleep(2.0)
    outputPath = videoPath.split(".")[0] + "OUT.avi"

    out = cv2.VideoWriter(outputPath,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (width, height))
    # keep looping
    while True:
        # grab the current frame
        if videoStream:
            ret, frame = vs.read()
        else:
            ret, frame = vs.read()
            # frame = frame[0:frame.shape[0], 300:(frame.shape[1] - 300)]
            # print(frame.shape)
            # frame = ret, frame

        cv2.namedWindow('Frame')

        # handle the frame from VideoCapture or VideoStream
        # frame = frame[1] if args.get("video", False) else frame

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video

        if frame is None:
            break

        # Contours here
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        # Threshold the YCrCb image to get only blue colors
        if colourMask == "B":
            mask = cv2.inRange(ycbcr, lower_blue, upper_blue)
        elif colourMask == "Y":
            # Threshold the YCrCb image to get only yellow colors
            mask = cv2.inRange(ycbcr, lower_yellow, upper_yellow)
        elif colourMask == "G":
            # Threshold the YCrCb image to get only green colors
            mask = cv2.inRange(ycbcr, lower_green, upper_green)
        elif colourMask == "R":
            # Threshold the YCrCb image to get only red colors
            mask = cv2.inRange(ycbcr, lower_red, upper_red)
        else:
            # Threshold the YCrCb image to get only red colors
            mask = cv2.inRange(ycbcr, lower_black, upper_black)

        mask = cv2.erode(mask, None, iterations=3)

        mask = cv2.dilate(mask, None, iterations=3)

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
            # center = (int(x), int(y))
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if 50 < radius < 150:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # update the points queue
                pts.insert(0, center)
                # loop over the set of tracked points
                if len(pts) == 1:
                    displacement_x = []
                    displacement_y = []
                    time_x = []
                    time_y = []
                    diameter = radius * 2
                    start_y, start_x = pts[0]
                elif len(pts) % 3 == 0:
                    displacement_x_calculated = ((start_x - pts[0][1]) * weight_diameter) / diameter
                    displacement_y_calculated = ((pts[0][0] - start_y) * weight_diameter) / diameter
                    displacement_x.append(displacement_x_calculated)
                    time_x.append(len(pts) / 25)
                    # if (displacement_y_calculated >= 0.1) or (displacement_y_calculated <= -0.1):
                    #     winsound.Beep(frequency, duration)
                    displacement_y.append(displacement_y_calculated)
                    time_y.append(len(pts) / 25)

            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # draw the connecting lines
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

        # HoughCircles code here (Worse than contours)

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

        out.write(frame)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(15 * (not pause_playback)) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
        elif key == ord("r"):
            out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (width, height))
            pts = []
        elif key == ord(" "):
            pause_playback = not pause_playback

    vs.release()

    # close all windows
    cv2.destroyAllWindows()

    # open video in local video player
    # startfile(outputPath)

    plt.subplot(321)
    plt.plot(time_x, displacement_x)
    plt.subplot(322)
    plt.plot(time_y, displacement_y)
    new_velocity_x = np.gradient(displacement_x, 0.5)
    new_velocity_y = np.gradient(displacement_y, 0.5)
    acceleration_x = np.gradient(new_velocity_x, 0.5)
    acceleration_y = np.gradient(new_velocity_y, 0.5)
    energy_x = []
    energy_y = []
    for i in range(0, len(new_velocity_x)):
        if acceleration_x[i] >= 0:
            energy_x.append(0.5 * 174 * (new_velocity_x[i] ** 2))
        else:
            energy_x.append(0)
        if acceleration_y[i] >= 0:
            energy_y.append(0.5 * 174 * (new_velocity_y[i] ** 2))
        else:
            energy_y.append(0)
    plt.subplot(323)
    plt.plot(time_x, new_velocity_x)
    plt.subplot(324)
    plt.plot(time_y, new_velocity_y)
    plt.subplot(325)
    plt.plot(time_x, energy_x)
    plt.subplot(326)
    plt.plot(time_y, energy_y)
    plt.show()


if __name__ == '__main__':
    # inPath = input("Enter the path to the video, or 0 to use camera: ")
    # diskColour = input("Enter a disk color, (B)lue, Blac(K), (Y)ellow, (G)reen, (R)ed: ")
    # if inPath == "0":
    #     inPath = int(inPath)
    inPath = "Videos/Ecem/Ecem8.mp4"
    diskColour = "Y"
    track(inPath, diskColour)
