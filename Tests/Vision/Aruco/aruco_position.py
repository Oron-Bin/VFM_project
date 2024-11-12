import os
import sys
import datetime
import time
import cv2
import csv

sys.path.insert(1, r'/')

from Utils.Control.cardalgo import *

# initial_flag = 0

# This code is used to record a video
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# filename = f"/home/roblab20/Desktop/article_videos/full_vibration/oron_{timestamp}.XVID"
VIDEO_DIR = "/home/roblab20/Desktop/article_videos/full_vibration"
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'XVID')

start_time = time.time()

CSV_FILE_PATH = "/home/roblab20/Desktop/article_videos/data_full_vibration"
csv_file_path = os.path.join(CSV_FILE_PATH, f"data_{timestamp}.csv")
# Create CSV file and write headers
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['angle'])
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
def get_dims(cap, res='480p'):
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
# out = cv2.VideoWriter(filename, get_video_type(filename), 7, get_dims(cam, res))
video_path = os.path.join(VIDEO_DIR, f"output_{timestamp}.avi")
out = cv2.VideoWriter(video_path, VIDEO_FOURCC, 20.0, (640, 480))
cam.set(3, 640)
cam.set(4, 480)

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
algo.x_d = random.randint(300, 360)  # set a random value of the goal x position
algo.y_d = random.randint(90, 140)  # set a random value of the goal y position
goal_position = [algo.x_d,algo.y_d]
algo.orientation = random.randint(0,359)
angle_teta = 0

oron = True
if state == 'calibrate' :
    mycard.calibrate()
    print('the system is ready to vibrate')
    # time.sleep(3)
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
            motor_flag = 2


        if circle_radius is not None and state == 'after_calibrate' and motor_flag ==2:
            if algo.card_initialize_pos(circle_center) == 1:
                print('hello world')
                state = 'vibrating'


          # now the vibration is continuous.
        if circle_radius is not None and state == 'vibrating' and motor_flag ==2:
            mycard.start_hardware()
            mycard.vibrate_hardware(100)
            algo.update(circle_center)
            algo.plot_desired_position(img)
            algo.plot_path(img)


            if algo.check_distance(epsilon=10) is False and circle_center is not None and state == 'vibrating' and motor_flag ==2:
                angle_teta = algo.calculate_angle(goal_position,circle_center)
                output = algo.law_1(first=oron)
                oron=False
                # output = 0
                delta_list.append(output)
                mycard.set_encoder_angle(output) ## Update the motor output
                algo.plot_arrow(img) ## Plot the direction of the motor
                # mycard.send_data('encoder') ## still dont sure if its necessary
                # time.sleep(0.001)

            if algo.check_distance(epsilon=10) is True and motor_flag == 2:
                state = 'stop_vibrating'
                if state == 'stop_vibrating':
                    # mycard.vibrate_hardware(0)
                    mycard.stop_hardware()
                    print('arrive')

                    # algo.next_iteration()
                    # j = j + 1
                    # algo.package_data()
                    # algo.clear()
                    # algo.random_input()
                    state = 'vibrating'
                    motor_flag = 2

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 'Orientation Error', 'Distance Error', 'Vibration Amp (%)'
            writer.writerow(
                [angle_teta])

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