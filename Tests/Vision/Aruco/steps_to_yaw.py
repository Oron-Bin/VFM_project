import numpy as np
from math import *
import matplotlib.pyplot as plt
import math
from matplotlib.path import Path
import matplotlib.patches as patches
import os
import time
import sys
import random
sys.path.insert(1,r'C:\Users\USER\Desktop\Card Control')

# from package import *
from Utils.Control.cardalgo import *

initial_flag = 0


"""This Code is use to record a video"""

filename = 'video.avi'
frames_per_second = 60
res = '720p'


# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


"""This part resposible for the close loop control using CV2 circle detection"""

cam = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cam, res))
cam.set(3,1280)
cam.set(4,720)
cam.set(cv2.CAP_PROP_AUTOFOCUS,0)
set_des = 0
mycard = Card(0,0,-1,-1,-1,-1,115200,'/dev/ttyACM0')
mycard.set_motor_angle(0.0001) ## it was 0.0001
mycard.send_data('motor')
algo = card_algorithms(0,0)
# while(1):
#     mycard.set_encoder_angle(algo.output_calibrate()) ## algo.output_calibrate()
#     mycard.send_data('encoder')


flag = 0

while cam.isOpened():
    center, Img = algo.filter_camera(cam, 3)
    aruco = algo.find_Aruco(Img)
    algo.finger_position(Img)
    # Capturing each frame of our video stream
    if set_des == 0: ## If user didnt input a value yeti
        algo.position_user_input(Img)
        set_des = 1
    elif center is not None and set_des == 1: ## If the first card_center is not updated yet
        algo.card_initialize(center[:-1])
        set_des = 2
        mycard.vibrate_on()
    elif center is not None:
        algo.desired_position(Img)
        # print(center)
        # print(center.tolist())
        algo.update(center.tolist())
        algo.plot_path(Img)
        if algo.check_distance(5) is not True and set_des == 2:
            output = algo.law_1()
            mycard.set_encoder_angle(output)
            algo.plot_arrow(Img)
            mycard.send_data('encoder')
        elif algo.check_distance(5) is True:
            for i in range(10):
                mycard.send_data('vibrate')
                set_des = 3
        if set_des == 3:
            time.sleep(1)

            for i in range(10):
                mycard.send_data('st')

            time.sleep(1)
            algo.next_iteration()
            algo.package_data()

            ## if we want throw center of mass

            if flag == 1:
                algo.random_input()
                flag = 0
            else:
                algo.y_d = 312
                algo.x_d = 624
                flag = 1

            ## if we want not threw center of mass

            # algo.random_input()
            algo.clear()
            if algo.iteration == 10:
                algo.export_data()
                break

            set_des = 2
        # time.sleep(0.1)
        algo.draw_circle(Img, center)
        # print(output)
    out.write(Img)
    cv2.imshow('QueryImage', Img)



    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Interupt by user')
        break
    if cv2.waitKey(1) & 0xFF == ord('i'):
        algo.position_user_input(Img)
#
# cv2.destroyAllWindows()
# plt.show()