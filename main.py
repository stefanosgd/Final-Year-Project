# import the necessary packages
from collections import deque
import numpy as np
import cv2
import imutils
import time
from tkinter import filedialog
import tkinter
from PIL import Image, ImageTk

from imutils.video import FPS
from matplotlib import pyplot as plt
import winsound
from os import startfile


def track(videoPath, colourMask):
    selectedTracker = "kcf"
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(selectedTracker)
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # appropriate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects

        tracker = OPENCV_OBJECT_TRACKERS[selectedTracker]()
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None

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
    line_break = 0

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

    # initialize the FPS throughput estimator
    fps = None

    # allow the camera or video file to warm up
    time.sleep(2.0)
    outputPath = videoPath.split(".")[0] + "OUT.mp4"
    cv2.namedWindow('Frame')


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

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video

        if frame is None:
            break

        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # cv2.circle(frame, (x+int(w/2), y+int(h/2)), int(w/2), (0, 255, 0), 2)
                (center, radius) = (x+int(w/2), y+int(h/2)), int(w/2)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                # cv2.circle(frame, center, int(radius),
                #            (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # update the points queue
                pts.insert(0, center)
                # loop over the set of tracked points
                if len(pts) == 1:
                    displacement_x = []
                    displacement_y = []
                    time_axis = []
                    diameter = radius * 2
                    start_y, start_x = pts[0]
                elif len(pts) % 3 == 0:
                    displacement_x_calculated = ((start_x - pts[0][1]) * weight_diameter) / diameter
                    displacement_y_calculated = ((pts[0][0] - start_y) * weight_diameter) / diameter
                    displacement_x.append(displacement_x_calculated)
                    time_axis.append(len(pts) / 30)
                    # if (displacement_y_calculated >= 0.1) or (displacement_y_calculated <= -0.1):
                    #     winsound.Beep(frequency, duration)
                    displacement_y.append(displacement_y_calculated)

                for i in range(1, len(pts)-line_break):
                    # if either of the tracked points are None, ignore
                    if pts[i - 1] is None or pts[i] is None:
                        continue

                    # draw the connecting lines
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", selectedTracker),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if initBB is None:
            out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (W, H))
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
                    initBB = cv2.boundingRect(c)
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    # cv2.rectangle(frame,(int(x-radius),int(y-radius)),(int(x+radius),int(y+radius)),(0,255,0),2)
                    # initBB = cv2.rectangle(frame,(int(x-radius),int(y-radius)),(int(x+radius),int(y+radius)),(0,255,0),2)
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    tracker.init(frame, initBB)
                    fps = FPS().start()
                    # pause_playback = not pause_playback

        out.write(frame)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(10 * (pause_playback)) & 0xFF

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("r"):
            if pause_playback:
                pause_playback = not pause_playback
            tracker = OPENCV_OBJECT_TRACKERS[selectedTracker]()
            initBB = None
            pts = []
        if key == ord("c"):
            line_break = len(pts)
        # if the 'q' key is pressed, stop the loop
        elif key == ord("s"):
            tracker = OPENCV_OBJECT_TRACKERS[selectedTracker]()
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                   showCrosshair=True)
            pts = []
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
            if not pause_playback:
                pause_playback = not pause_playback
        elif key == ord(" "):
            pause_playback = not pause_playback
        elif key == ord("q"):
            break

    vs.release()
    out.release()

    # close all windows
    cv2.destroyAllWindows()

    # open video in local video player
    # startfile(outputPath)

    plt.subplot(321)
    plt.plot(time_axis, displacement_x)
    plt.subplot(322)
    plt.plot(time_axis, displacement_y)
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
    plt.plot(time_axis, new_velocity_x)
    plt.subplot(324)
    plt.plot(time_axis, new_velocity_y)
    plt.subplot(325)
    plt.plot(time_axis, energy_x)
    plt.subplot(326)
    plt.plot(time_axis, energy_y)
    plt.show()


def initialise_gui():

    def file_select():
        file = tkinter.filedialog.askopenfilename(initialdir="./Videos", title="Select File",
                                                  filetypes=[("Video", "*.MOV;*.MP4;*.AVI")])
        if len(file) > 0:
            first_frame = None
            file_location.set(file)
            path_label.config(text=("The selected file is:\n " + file_location.get()))
            path_label.grid(row=2, column=2, rowspan=2)
            # Run Button
            tkinter.Button(top, text="Track", command=start_tracking).grid(row=4, column=2)

            vs = cv2.VideoCapture(file_location.get())
            _, frame = vs.read()
            frame = imutils.resize(frame, width=300)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            if first_frame is None:
                first_frame = tkinter.Label(image=image)
                first_frame.image = image
                first_frame.grid(row=1, column=3, rowspan=7)
            else:
                first_frame.configure(image=image)
                first_frame.image = image

            vs.release()

    def start_tracking():
        track(file_location.get(), colour_choice.get())

    top = tkinter.Tk()
    colour_choice = tkinter.StringVar(value="K")
    file_location = tkinter.StringVar(value="")
    # widgets start here
    tkinter.Label(top, text="Welcome To OW Tracker", font=("Ariel", 50)).grid(row=0, column=0, columnspan=5)
    path_label = tkinter.Label(top)
    # Weight Colour Selection
    tkinter.Label(top, text="What colour weights are you using?").grid(row=1, column=0, columnspan=1)
    tkinter.Radiobutton(top, text="Black", variable=colour_choice, value="K").grid(padx=75, row=2, column=0, sticky="W")
    tkinter.Radiobutton(top, text="Blue", variable=colour_choice, value="B").grid(padx=75, row=3, column=0, sticky="W")
    tkinter.Radiobutton(top, text="Yellow", variable=colour_choice, value="Y").grid(padx=75, row=4, column=0, sticky="W")
    tkinter.Radiobutton(top, text="Green", variable=colour_choice, value="G").grid(padx=75, row=5, column=0, sticky="W")
    tkinter.Radiobutton(top, text="Red", variable=colour_choice, value="R").grid(padx=75, row=6, column=0, sticky="W")

    # File Selection Button
    tkinter.Button(top, text="Select a video file", command=file_select).grid(row=1, column=2)

    # widgets end here
    top.mainloop()


if __name__ == '__main__':
    initialise_gui()
    # inPath = input("Enter the path to the video, or 0 to use camera: ")
    # diskColour = input("Enter a disk color, (B)lue, Blac(K), (Y)ellow, (G)reen, (R)ed: ")
    # if inPath == "0":
    #     inPath = int(inPath)
    # track(inPath, diskColour)
    # inPath = "Videos/Sophie/Sophie1.mp4"
    # diskColour = "K"
    # track(inPath, diskColour)
    #
    # inPath = "Videos/Sophie/Sophie2.mp4"
    # diskColour = "Y"
    # track(inPath, diskColour)
    #
    # inPath = "Videos/Sophie/Sophie3.mp4"
    # diskColour = "K"
    # track(inPath, diskColour)
    #
    # inPath = "Videos/Sophie/Sophie4.mp4"
    # diskColour = "K"
    # track(inPath, diskColour)
    #
    # inPath = "Videos/Sophie/Sophie5.mp4"
    # diskColour = "K"
    # track(inPath, diskColour)
    #
    # inPath = "Videos/Sophie/Sophie6.mp4"
    # diskColour = "K"
    # track(inPath, diskColour)
    #
    # inPath = "Videos/Sophie/Sophie7.mp4"
    # diskColour = "K"
    # track(inPath, diskColour)


