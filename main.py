# import the necessary packages
import numpy as np
import cv2
import imutils
import time
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from imutils.video import FPS
from matplotlib import pyplot as plt
import winsound
from os import startfile


def track(videoPath, colourMask, totalWeight):
    new_path = True
    total_frames = 0
    dropped_frames = 0
    selectedTracker = "mosse"

    # initialize a dictionary that maps strings to their corresponding OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
        "goturn": cv2.TrackerGOTURN_create
    }

    # grab the appropriate object tracker using our dictionary of OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[selectedTracker]()

    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None

    pause_playback = False  # pause until key press after each image

    upper_black = np.array([100, 135, 155])
    lower_black = np.array([10, 110, 120])

    # define range of blue color in YCrCb
    upper_blue = np.array([60, 130, 150])
    lower_blue = np.array([20, 115, 130])

    # define range of yellow color in YCrCb
    upper_yellow = np.array([255, 170, 90])
    lower_yellow = np.array([0, 50, 10])

    # define range of green color in YCrCb
    upper_green = np.array([255, 110, 130])
    lower_green = np.array([0, 10, 10])

    # define range of red color in YCrCb
    upper_red = np.array([255, 131, 176])
    lower_red = np.array([100, 111, 153])

    weight_diameter = 450 / 1000  # mm
    start_x = 0
    start_y = 0
    line_break = 0
    displacement_x = [0]
    displacement_y = [0]
    time_axis = [0]

    # initialize the list of tracked points, the frame counter,
    # and the coordinate deltas
    pts = []
    # if a video path was not supplied, grab the reference
    # to the webcam
    if videoPath == 0:
        vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        videoStream = True
        outputPath = "./Videos/Live/LiveOUT.mp4"
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(videoPath)
        videoStream = False
        outputPath = videoPath.split(".")[0] + "OUT.mp4"

    # initialize the FPS throughput estimator
    fps = None

    # allow the camera or video file to warm up
    time.sleep(2.0)
    cv2.namedWindow('Frame')

    # keep looping
    while True:
        # grab the current frame
        if videoStream:
            ret, frame = vs.read()
        else:
            ret, frame = vs.read()

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video

        if frame is None:
            break

        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        clean_frame = frame.copy()
        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # cv2.circle(frame, (x+int(w/2), y+int(h/2)), int(w/2), (0, 255, 0), 2)
                (center, diameter) = (x + int(w / 2), y + int(h / 2)), int(w)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                # cv2.circle(frame, center, int(radius),
                #            (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.line(frame, start_center, start_center_top, (0, 255, 0), 3)

                # update the points queue
                pts.insert(0, center)
                # loop over the set of tracked points
                if len(pts) == 1:
                    start_y, start_x = pts[0]
                    # diameter = radius * 2
                elif len(pts) % 3 == 0:
                    displacement_x_calculated = ((start_x - pts[0][1]) * weight_diameter) / diameter
                    displacement_y_calculated = ((pts[0][0] - start_y) * weight_diameter) / diameter
                    displacement_x.append(displacement_x_calculated)
                    time_axis.append(len(pts) / 30)
                    if ((displacement_y_calculated >= 0.4) or (displacement_y_calculated <= -0.2)) and videoStream:
                        winsound.Beep(1000, 100)
                    displacement_y.append(displacement_y_calculated)

                for i in range(1, len(pts) - line_break):
                    # if either of the tracked points are None, ignore
                    if pts[i - 1] is None or pts[i] is None:
                        continue

                    # draw the connecting lines
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 3)
            else:
                dropped_frames += 1
            total_frames += 1
            # update the FPS counter

            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", selectedTracker),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
                ("Weight", "{} KG".format(totalWeight)),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (W, H))
            # Contours here
            ycrcb = cv2.cvtColor(cv2.blur(frame, (3, 3)), cv2.COLOR_BGR2YCR_CB)

            # Threshold the YCrCb image to get only blue colors
            if colourMask == "B":
                mask = cv2.inRange(ycrcb, lower_blue, upper_blue)
            elif colourMask == "Y":
                # Threshold the YCrCb image to get only yellow colors
                mask = cv2.inRange(ycrcb, lower_yellow, upper_yellow)
            elif colourMask == "G":
                # Threshold the YCrCb image to get only green colors
                mask = cv2.inRange(ycrcb, lower_green, upper_green)
            elif colourMask == "R":
                # Threshold the YCrCb image to get only red colors
                mask = cv2.inRange(ycrcb, lower_red, upper_red)
            else:
                # Threshold the YCrCb image to get only black colors
                mask = cv2.inRange(ycrcb, lower_black, upper_black)

            mask = cv2.erode(mask, None, iterations=4)

            mask = cv2.dilate(mask, None, iterations=3)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            # only processed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                start_center = (int(x), int(y))
                start_center_top = (int(x), int(y)-600)
                # only proceed if the radius meets a minimum size
                if 50 < radius < 250:
                    initBB = cv2.boundingRect(c)
                    # draw the circle and centroid on the frame,
                    # then initialise the tracker
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, start_center, 5, (0, 0, 255), -1)
                    cv2.line(frame, start_center, start_center_top, (0, 255, 0), 3)
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    tracker.init(frame, initBB)
                    fps = FPS().start()

        # show the output frame
        out.write(frame)
        cv2.imshow("Frame", frame)
        # cv2.imshow("Mask", mask)
        key = cv2.waitKey(1 * pause_playback) & 0xFF

        # if the 'r' key is selected, we are going to "select" a bounding box to track
        if key == ord("r"):
            if pause_playback:
                pause_playback = not pause_playback
            tracker = OPENCV_OBJECT_TRACKERS[selectedTracker]()
            initBB = None
            pts = []
            displacement_x = [0]
            displacement_y = [0]
            time_axis = [0]
        if key == ord("a"):
            line_break = len(pts)
        # if the 's' key is selected, we are going to select a bounding box to track
        elif key == ord("s"):
            tracker = OPENCV_OBJECT_TRACKERS[selectedTracker]()
            # create a text trap and redirect stdout
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", clean_frame, fromCenter=False,
                                   showCrosshair=True)
            start_center = (initBB[0] + int(initBB[2]/2), initBB[1] + int(initBB[3]/2))
            start_center_top = (initBB[0] + int(initBB[2]/2), initBB[1] + int(initBB[3]/2) - 600)
            cv2.line(frame, start_center, start_center_top, (0, 255, 0), 3)
            pts = []
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()

            if not pause_playback:
                pause_playback = not pause_playback
        elif key == ord(" "):
            pause_playback = not pause_playback
        # if the 'q' key is pressed, stop the loop
        elif key == ord("q"):
            break

    vs.release()
    out.release()

    # close all windows
    cv2.destroyAllWindows()

    # todo delete
    # output the accuracy
    # print("Total frames: {}".format(total_frames))
    # print("Dropped frames: {}".format(dropped_frames))
    # print("Accuracy: {}".format((total_frames - dropped_frames) / total_frames))

    # open video in local video player
    # startfile(outputPath)

    if len(time_axis) > 1:
        plt.figure(figsize=(15, 8))
        plt.subplot(321)
        plt.grid()
        plt.title("Vertical vs Time")
        plt.ylabel("Displacement (M)")
        plt.plot(time_axis, displacement_x)
        plt.subplot(322)
        plt.grid()
        plt.title("Horizontal vs Time")
        plt.ylabel("Displacement (M)")
        plt.plot(time_axis, displacement_y)

        new_velocity_x = np.gradient(displacement_x, time_axis)  # was -> , 0.5)
        new_velocity_y = np.gradient(displacement_y, time_axis)
        acceleration_x = np.gradient(new_velocity_x, time_axis)
        acceleration_y = np.gradient(new_velocity_y, time_axis)
        energy_x = []
        energy_y = []
        for i in range(0, len(new_velocity_x)):
            if (acceleration_x[i] >= 0) and ((displacement_x[i] >= 0.1) or (new_velocity_x[i] >= 0)):
                energy_x.append(0.5 * totalWeight * (new_velocity_x[i] ** 2))
            else:
                energy_x.append(0)
            if acceleration_y[i] >= 0:
                energy_y.append(0.5 * totalWeight * (new_velocity_y[i] ** 2))
            else:
                energy_y.append(0)
        plt.subplot(323)
        plt.grid()
        plt.ylabel("Velocity (M/s)")
        plt.plot(time_axis, new_velocity_x)
        plt.subplot(324)
        plt.grid()
        plt.ylabel("Velocity (M/s)")
        plt.plot(time_axis, new_velocity_y)
        plt.subplot(325)
        plt.grid()
        plt.ylabel("Energy (J)")
        plt.plot(time_axis, energy_x)
        plt.subplot(326)
        plt.grid()
        plt.ylabel("Energy (J)")
        plt.plot(time_axis, energy_y)
        plt.tight_layout(3.0)
        plt.show()


def initialise_gui():
    def file_select():
        file = tkinter.filedialog.askopenfilename(initialdir="./Videos", title="Select File",
                                                  filetypes=[("Video", "*.MOV;*.MP4;*.AVI")])
        if len(file) > 0:
            file_location.set(file)
            path_label.config(text=(file_location.get().split("/")[-1]), font=("Ariel", 10))
            path_label.grid(row=2, column=1, padx=150, sticky="NW", rowspan=5)
            # Run Button
            tkinter.Button(top, text="Track", command=start_tracking, height=1, width=10, font=("Ariel", 25)).grid(
                row=8, column=1)

            vs = cv2.VideoCapture(file_location.get())
            _, frame = vs.read()
            frame = imutils.resize(frame, width=200)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            first_frame.configure(image=image)
            first_frame.image = image
            first_frame.grid(row=2, column=1, rowspan=5, sticky="S")

            vs.release()
            cv2.destroyAllWindows()

    def start_tracking():
        try:
            weight = int(weight_used.get())
            tkinter.messagebox.showinfo("Directions", "Use 'Space' to start tracking.\n"
                                                      "Use 'S' to select a new area to track. Use your mouse to draw "
                                                      "the bounding box around the plate, and then 'Space' to start "
                                                      "tracking\n"
                                                      "Use 'R' to reset the tracking and attempt to find the plates\n"
                                                      "Use 'A' to clear the current path for multiple reps\n"
                                                      "Use 'Q' to quit")
            track(file_location.get(), colour_choice.get(), weight)
        except ValueError:
            tkinter.messagebox.showerror("Error", "Please input a valid Integer for the weight used")

    def live_tracking():
        try:
            weight = int(weight_used.get())
            tkinter.messagebox.showinfo("Directions", "Use 'Space' to start tracking.\n"
                                                      "Use 'S' to select a new area to track. Use your mouse to draw "
                                                      "the bounding box around the plate, and then 'Space' to start "
                                                      "tracking\n"
                                                      "Use 'R' to reset the tracking and attempt to find the plates\n"
                                                      "Use 'A' to clear the current path for multiple reps\n"
                                                      "Use 'Q' to quit")
            track(0, colour_choice.get(), weight)
        except ValueError:
            tkinter.messagebox.showerror("Error", "Please input a valid Integer for the weight used")

    top = tkinter.Tk()
    top.geometry("575x600")
    colour_choice = tkinter.StringVar(value="K")
    file_location = tkinter.StringVar(value="")
    first_frame = tkinter.Label(image=None)

    # widgets start here
    tkinter.Label(top, text="Welcome To OW Tracker", font=("Ariel", 35)).grid(padx=20, row=0, column=0, columnspan=7,
                                                                              sticky="W")

    path_label = tkinter.Label(top, wraplength=70)
    # Weight Colour Selection
    tkinter.Label(top, text="Weights Colour", font=("Ariel", 20)).grid(padx=25, row=1, column=0, columnspan=2,
                                                                       sticky="W")
    tkinter.Radiobutton(top, text="Black", variable=colour_choice, value="K", font=("Ariel", 20)).grid(padx=55, pady=15,
                                                                                                       row=2, column=0,
                                                                                                       sticky="W")
    tkinter.Radiobutton(top, text="Blue", variable=colour_choice, value="B", font=("Ariel", 20)).grid(padx=55, pady=15,
                                                                                                      row=3, column=0,
                                                                                                      sticky="W")
    tkinter.Radiobutton(top, text="Yellow", variable=colour_choice, value="Y", font=("Ariel", 20)).grid(padx=55,
                                                                                                        pady=15, row=4,
                                                                                                        column=0,
                                                                                                        sticky="W")
    tkinter.Radiobutton(top, text="Green", variable=colour_choice, value="G", font=("Ariel", 20)).grid(padx=55, pady=15,
                                                                                                       row=5, column=0,
                                                                                                       sticky="W")
    tkinter.Radiobutton(top, text="Red", variable=colour_choice, value="R", font=("Ariel", 20)).grid(padx=55, pady=15,
                                                                                                     row=6, column=0,
                                                                                                     sticky="W")

    # Weight input
    tkinter.Label(top, text="Enter the weight of your bar (KG): ", font=("Ariel", 15)).grid(padx=20, row=7, column=0, columnspan=2, sticky="W")
    weight_used = tkinter.Entry(top)
    weight_used.grid(row=7, column=1)
    # File Selection Button
    tkinter.Button(top, text="Select a video file", command=file_select, height=1, width=15, font=("Ariel", 13)).grid(
        padx=114, row=1, column=1)
    tkinter.Button(top, text="Live", command=live_tracking, height=1, width=10, font=("Ariel", 25)).grid(row=8,
                                                                                                         column=0)
    # widgets end here
    top.mainloop()


if __name__ == '__main__':
    initialise_gui()
