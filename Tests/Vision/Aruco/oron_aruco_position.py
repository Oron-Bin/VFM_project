import os
import sys
import datetime
import time
import cv2

sys.path.insert(1, r'/')

from Utils.Control.cardalgo import *

# initial_flag = 0

# This code is used to record a video
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"/home/roblab20/Desktop/videos/oron_videos/oron_{timestamp}.avi"
start_time = time.time()

# Define video resolution
res = '720p'
# data_list = []


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
state = 'calibrate' #the default is that we need to calibrate the system first
scale = 28 # 1 cm = 28 pixels
j = 0
orientation_list = []
delta_list = []
motor_flag  = 0
vib_flag = True

start = time.perf_counter()
algo.x_d = 600  # set a random value of the goal x position
algo.y_d = 210  # set a random value of the goal y position
goal_position = [algo.x_d,algo.y_d]
algo.orientation = random.randint(0,359)
angle_teta = 0


if state == 'calibrate' :
    mycard.calibrate()
    print('the system is ready to vibrate')
    time.sleep(3)
    state = 'after_calibrate'


while cam.isOpened():

    ret, img = cam.read()
    time_diff = time.time() - start_time

    if ret:

        circle_center, circle_radius = algo.detect_circle_info(img)
        origin = tuple(algo.finger_position(img))
        aruco_centers, ids = algo.detect_aruco_centers(img)
        start = time.perf_counter()

        if motor_flag == 0:
            #
            # mycard.start_hardware()
            # mycard.set_encoder_angle(-45)
            motor_flag = 2


        if circle_radius is not None and state == 'after_calibrate' and motor_flag ==2:
            if algo.card_initialize(circle_center) == 1:
                print('hello world')
                state = 'vibrating'


          # now the vibration is continuous.
        if circle_radius is not None and state == 'vibrating' and motor_flag ==2:
            mycard.start_hardware()
            mycard.vibrate_hardware(50)
            algo.update(circle_center)
            algo.plot_desired_position(img)
            algo.plot_path(img)
            if ids is not None and len(ids) > 0:
                orientation_angle = algo.ids_to_angle(ids, circle_center, aruco_centers)
                orientation_error = abs(algo.shortest_way(orientation_angle, algo.orientation))
                print(orientation_error)
                cv2.putText(img, f"error: {round(orientation_error, 1)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if orientation_error < 10 :
                mycard.stop_hardware()
                print('the orientation error is: {0}'.format(orientation_error))
                state = 10
                if state == 10 :
                    mycard.calibrate()
                    time.sleep(5)
                    mycard.start_hardware()

                    mycard.set_encoder_angle(30) #for example the angle is 30
                    state = 11

            if algo.check_distance(epsilon=10) is False and circle_center is not None and state == 'vibrating' and motor_flag ==2:
                angle_teta = algo.calculate_angle(goal_position,circle_center)
                output = algo.law_1()
                # output = 0
                delta_list.append(output)
                mycard.set_encoder_angle(output) ## Update the motor output
                algo.plot_arrow(img) ## Plot the direction of the motor
                # mycard.send_data('encoder') ## still dont sure if its necessary
                # time.sleep(0.001)

            if algo.check_distance(epsilon=10) is True:
                state = 'stop_vibrating'
                if state == 'stop_vibrating':
                    mycard.stop_hardware()
                    print('arrive')
                    algo.next_iteration()
                    j = j + 1
                    algo.package_data()
                    algo.clear()
                    algo.random_input()
                    state = 'vibrating'
                    motor_flag = 0



        cv2.putText(img, f"motor_angle: {angle_teta}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(img)  # Saves the current frame to the video file.
        algo.display_image(img, circle_center,circle_radius)  # shows the marker circle center and circle shape in red color
        cv2.imshow('QueryImage', img)  # shows the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interrupted by user')
            # mycard.stop_hardware()
            break
        if cv2.waitKey(1) & 0xFF == ord('i'):
            algo.position_user_input(img)


cam.release()
out.release()
cv2.destroyAllWindows()

