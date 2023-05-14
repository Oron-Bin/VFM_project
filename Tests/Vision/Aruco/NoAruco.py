import os
import sys

sys.path.insert(1, r'/')

from Utils.Control.cardalgo import *

initial_flag = 0


"""This Code is use to record a video"""

filename = '../../../../video.avi'
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


"""This part responsible for the close loop control using CV2 circle detection"""
#==========================
#Defines camera parameters#
#==========================
cam = cv2.VideoCapture(0)
# out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cam, res))
cam.set(3,1280)
cam.set(4,720)
# cam.set(cv2.CAP_PROP_AUTOFOCUS,0)

## Set the card class and open the serial communication
mycard = Card(x_d=0,y_d=0,a_d=-1,x=-1,y=-1,a=-1,baud=115200,port='/dev/ttyACM0')
mycard.set_motor_angle(0.0001) ## it was 0.0001 ## Update the motor angle value
mycard.send_data(key='motor') ## Send data to the motor/ this func in package
algo = card_algorithms(x_d=0,y_d=0) # Define the card algorithm object


# Define the set desired paramter and tell the code that user didnt init it yet
set_des = 0

# while(1):
#     mycard.set_encoder_angle(algo.output_calibrate()) ## algo.output_calibrate()
#     mycard.send_data('encoder')


flag = 0



j = 0
while cam.isOpened():

    center, Img = algo.filter_camera(cam=cam, filter=3) ## Update the card center
    # print(tuple(center))
    # orientation = algo.find_card_orientation(Img) ## Update the orientation
    # print(orientation) #for me to check if the cam find and update the orientaion
    algo.finger_position(Img,calibration=False) ## If Main axis system need calibartion change to True and calibrate the xy point
    # algo.plot_desired_path(Img,(-30,30),(-30,30)) ##TODO check this what its doing?
    # Capturing each frame of our video stream

    if set_des == 0: ## If user didnt input a value yet
        print("No des yet")
        ##############################
        #########Generate Structure paths############
        #
        # rectangle = algo.generate_path()
        # circle = algo.generate_circle()
        # heart = algo.generate_heart()

        ##############################
        ######For User input#########

        # algo.position_user_input()
        algo.y_d = 227 ## 220
        algo.x_d = 668
        start = time.perf_counter()
        ##############################

        #######################################
        ######For Rectangle Path Tracking######
        #
        # algo.x_d = rectangle[0][0]
        # print(algo.x_d)
        # algo.y_d = rectangle[0][1]
        # print(algo.y_d)

        #########################################
        ############## For Circle Path Tracking ############

        # algo.x_d = circle[0][0]
        # print(algo.x_d)
        # algo.y_d = circle[0][1]
        # print(algo.y_d)

        #########################################
        ############## For Heart Path Tracking ############

        # algo.x_d = heart[0][0]
        # print(algo.x_d)
        # algo.y_d = heart[0][1]
        # print(algo.y_d)

        set_des = 1
        print(set_des)

    elif center is not None and set_des == 1: ## If the first card_center is not updated yet
        if (algo.card_initialize(center[:-1],)) == 1:
            set_des = 2
            mycard.vibrate_on()

    elif center is not None:
        algo.plot_desired_position(Img)

        algo.update(center.tolist())
        algo.plot_path(Img)
        # print(Img)
        # if j > 1:
        #     algo.plot_path(Img)
        # else:
        #     algo.clear()

        if algo.check_distance(epsilon=5) is not True and set_des == 2:

            ## If you want to choose control law number 1

            output = algo.law_1()

            ###############################################

            mycard.set_encoder_angle(output) ## Update the motor output
            algo.plot_arrow(Img) ## Plot the direction of motor
            mycard.send_data('encoder') ## Send the motor output to the hardware
            # print(start)
            # print(time.perf_counter())
            # if time.perf_counter() > start + 20:
            #     algo.package_data()
            #     algo.export_data()
            #     break
            time.sleep(0.1)
        elif algo.check_distance(5) is True:
            for i in range(10):
                mycard.send_data('vibrate')
                set_des = 3
        if set_des == 3:
            time.sleep(1) # a delay of a sec between each iteration

            for i in range(10):
                mycard.send_data('st') # or set des 3 or this is stop the vibration

            # time.sleep(1)
            algo.next_iteration()
            j = j + 1
            algo.package_data()

            #########################
            ###### Update heart path ########

            # if j == len(heart):
            #     algo.export_data()
            #     break
            # algo.x_d = heart[j][0]
            # algo.y_d = heart[j][1]

            #########################
            ###### Update Circle path ########

            # if j == len(circle):
            #     algo.export_data()
            #     break
            # algo.x_d = circle[j][0]
            # algo.y_d = circle[j][1]

            ##########################
            ##### Update the rectangle path #######

            # if j == len(rectangle):
            #     algo.export_data()
            #     break
            # algo.x_d = rectangle[j][0]
            # algo.y_d = rectangle[j][1]

            #########################################


            ## if we want through center of mass
            #############################

            # if flag == 1:
            #     print(algo.iteration)
            #     algo.package_data()
            #     # algo.clear()
            #     time.sleep(0.5)
            #     # algo.y_d = 240
            #     # algo.x_d = 624
            #     algo.random_input()
            #     flag = 0
            # else:
            #
            #     # algo.random_input()
            #     algo.y_d = 227
            #     algo.x_d = 668
            #     algo.clear()
            #     flag = 1

            ############################
            ## if we want not through center of mass
            ############################

            algo.clear()
            algo.random_input()

            ############################

            ############################
            ## if we want rectangle path tracking uncomment this


            # if algo.iteration == 10:
            #     algo.export_data()
            #     break

            set_des = 2
        # time.sleep(0.1)
        algo.draw_circle(Img, center)
        # print("output")
        print("set des is" , set_des)
    # out.write(Img)
    cv2.imshow('QueryImage', Img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Interupt by user')
        break
    if cv2.waitKey(1) & 0xFF == ord('i'):
        algo.position_user_input(Img)
#
cv2.destroyAllWindows()
plt.show()

