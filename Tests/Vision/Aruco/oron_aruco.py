import os
import sys
import datetime
import time
import cv2

sys.path.insert(1, r'/')

from Utils.Control.cardalgo import *

initial_flag = 0

# This code is used to record a video
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"/home/roblab20/Desktop/videos/oron_videos/oron_{timestamp}.avi"
start_time = time.time()

# Define video resolution
res = '720p'
data_list = []

# Utility function to get the shortest way between two angles
def shortest_way(num_1, num_2):
    if abs(num_1 - num_2) < 180:
        return num_2 - num_1
    else:
        if num_1 > num_2:
            return abs(num_1 - num_2 - 360)
        else:
            return abs(num_1 - num_2) - 360

# Function to change the resolution of the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Get the dimensions for the requested resolution
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

# Define video codec types
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

# Get the video type based on the file extension
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# Initialize camera parameters and video writer
cam = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 7, get_dims(cam, res))
cam.set(3, 1280)
cam.set(4, 720)

# Set the card class and open the serial communication
mycard = Card(x_d=0, y_d=0, a_d=-1, x=-1, y=-1, a=-1, baud=115200, port='/dev/ttyACM0')
algo = card_algorithms(x_d=0, y_d=0)  # Define the card algorithm object

# Define the set desired parameter and tell the code that the user didn't initialize it yet
state = 0 #the default is that we need to calibrate the system first
scale = 28 # 1 cm = 28 pixels





while cam.isOpened():
    ret, img = cam.read()
    time_diff = time.time() - start_time

    if state == 0:  # Calibrate the system
        mycard.calibrate()
        print('Calibration is done')
        algo.x_d = random.randint(500, 600)  # set a random value of the goal x position
        algo.y_d = random.randint(250, 300)  # set a random value of the goal y position

        state = 1  # Now the system is calibrated

        if state ==1 :
            mycard.start_hardware()
            mycard.vibrate_hardware(100)
            state = 2 #
        if state == 2:
            if algo.y_d is not None:
                mycard.stop_hardware()
    # state = 1
    if ret:
        out.write(img)  # Saves the current frame to the video file.
        algo.display_image(img, circle_center,circle_radius)  # shows the marker circle center and circle shape in red color
        cv2.imshow('QueryImage', img)  # shows the frame
        # state = 1
        circle_center, circle_radius = algo.detect_circle_info(img)
        # print('the COM is:', circle_center)
        # print('the radius is', circle_radius)
        aruco_centers, ids = algo.detect_aruco_centers(img)
        state = 3

        if circle_radius is not None and state ==3: #this line solve the problem of None type
            # print('the radius is', circle_radius/scale)
            # goal_pos = algo.plot_desired_position(img)
            origin = tuple(algo.finger_position(img))
            algo.plot_desired_position(img)
            # print(goal_pos, origin)


            # if state == 1 and goal_pos is not None:  # If the system is calibrated
            #     # origin = tuple(algo.finger_position(img)) #green point on screen
            #     print('hello world')
            #     mycard.start_hardware()
            #     mycard.vibrate_hardware(100)
            #     state = 2
            #     # if circle_center is not None: #this line solve the problem of None type
            #     # error = np.sqrt((circle_center[0]-goal_pos[0])**2 + (circle_radius[1]-goal_pos[1])**2)
            #     # print('the pos error is', error)
            # if state == 2:
            #         # error = np.sqrt((circle_center[0] - goal_pos[0]) ** 2 + (circle_radius[1] - goal_pos[1]) ** 2)
            #         # print('the pos error is', error)
            #         # if error < 10 :
            #         if algo.check_distance(epsilon=10) is True:
            #             print('this is enough')
            #             mycard.stop_hardware()



        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interrupted by user')
            # mycard.stop_hardware()
            break
        # if cv2.waitKey(1) & 0xFF == ord('i'):
        #     algo.position_user_input(img)


cam.release()
out.release()
cv2.destroyAllWindows()

