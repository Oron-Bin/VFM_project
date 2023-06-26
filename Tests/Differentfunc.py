import os
import sys
from datetime import datetime
import csv
import pandas as pd
import cv2
from collections import deque
import numpy as np
import time

sys.path.insert(1, r'/')

from Utils.Control.cardalgo import Card, card_algorithms

# Constants
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

# File paths
output_folder = "/home/roblab20/Desktop/videos/oron_videos/"
data_folder = "/home/roblab20/Desktop/videos/data_oron/"
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
video_filename = os.path.join(output_folder, f"oron_{timestamp}.avi")
csv_filename = os.path.join(data_folder, f"data_oron_{timestamp}.csv")

# Video settings
resolution = '480p'
frame_rate = 7

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

def get_video_type(filename):
    _, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# Define camera parameters
cam = cv2.VideoCapture(0)
out = cv2.VideoWriter(video_filename, get_video_type(video_filename), frame_rate, get_dims(cam, resolution))
cam.set(3, 1280)
cam.set(4, 720)

# Set up card and algorithm objects
mycard = Card(x_d=0, y_d=0, a_d=-1, x=-1, y=-1, a=-1, baud=115200, port='/dev/ttyACM0')
mycard.set_motor_angle(0.0001)
mycard.send_data(key='motor')
algo = card_algorithms(x_d=0, y_d=0)
set_des = False

# Data storage
df = pd.DataFrame(columns=['Orientation', 'Pos_x', 'pos_y', 'Motor angle', 'delta_teta', 'Time'])
orientation_list = deque()
delta_list = deque()

start_time = time.time()
# Main loop
while cam.isOpened():
    ret, img = cam.read()
    time_diff = time.time() - start_time

    if ret:
        circle_center, circle_radius = algo.detect_circle_info(img)
        aruco_centers, ids = algo.detect_aruco_centers(img)
        algo.finger_position(img, calibration=False)

        if not set_des:
            print("No desired position yet")
            algo.y_d = 227
            algo.x_d = 668
            start = time.perf_counter()
            print('The goal position is', algo.x_d, algo.y_d)
            set_des = True

        elif circle_center is not None and set_des:
            if algo.card_initialize(circle_center) == 1:
                set_des = True
                mycard.vibrate_on()

        elif circle_center is not None:
            algo.plot_desired_circle(img, algo.x_d, algo.y_d)
            algo.circle_tracking(img, circle_center, circle_radius, desired=(algo.x_d, algo.y_d))
            if algo.in_range(algo.x, algo.y, algo.x_d, algo.y_d, 5):
                mycard.vibrate_off()

        # algo.aruco_tracking(img, aruco_centers, ids)

        cv2.imshow('Robot Eye', img)
        out.write(img)

        if algo.a != -1 and algo.a_d != -1:
            orientation_list.append(algo.a)
            delta_list.append(algo.a_d)

        if len(orientation_list) > 100:
            orientation_list.popleft()
            delta_list.popleft()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Save data to CSV file
df['Orientation'] = orientation_list
df['delta_teta'] = delta_list
df['Time'] = time_diff
df.to_csv(csv_filename, index=False)

# Release resources
cam.release()
out.release()
cv2.destroyAllWindows()