import os
import sys
import datetime
import time
import cv2
import random
from Utils.Control.cardalgo import *

sys.path.insert(1, r'/')

# Video settings
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"/home/roblab20/Desktop/videos/main_orientation/oron_{timestamp}.avi"
res = '720p'

# Video resolution settings
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS.get(res, STD_DIMENSIONS["480p"])
    change_res(cap, width, height)
    return width, height

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    _, ext = os.path.splitext(filename)
    return VIDEO_TYPE.get(ext, VIDEO_TYPE['avi'])

# Initialize camera and video writer
cam = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 7, get_dims(cam, res))
cam.set(3, 1280)
cam.set(4, 720)

# Initialize card and algorithm
mycard = Card(x_d=0, y_d=0, a_d=-1, x=-1, y=-1, a=-1, baud=115200, port='/dev/ttyACM0')
algo = card_algorithms(x_d=0, y_d=0)
algo.orientation = random.randint(0, 359)
algo.x_d, algo.y_d = 644, 185

# Initial setup
state = 'calibrate'
scale = 28
orientation_list, delta_list = [], []
motor_flag = 0
vib_flag = True
flag = 0
first_time = 0
start_time = time.time()
oron = True

if state == 'calibrate' and motor_flag == 0:
    mycard.calibrate()
    print('the system is ready to vibrate')
    cv2.waitKey(3000)
    motor_flag = 1

while cam.isOpened():
    ret, img = cam.read()
    if not ret:
        break

    time_diff = time.time() - start_time
    circle_center, circle_radius = algo.detect_circle_info(img)
    print(f'circle_center: {circle_center}')

    if circle_radius is not None and algo.card_initialize(circle_center) == 1:
        print('hello world')
        state = 'after_calibrate'
        algo.update(circle_center)
        algo.plot_desired_position(img)
        algo.plot_path(img)

    aruco_centers, ids = algo.detect_aruco_centers(img)
    if ids is not None and len(ids) > 0 and state == 'after_calibrate':
        orientation_angle = algo.ids_to_angle(ids, circle_center, aruco_centers)
        orientation_error = abs(algo.shortest_way(orientation_angle, algo.orientation))
        cv2.putText(img, f"error: {round(orientation_error, 1)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if orientation_error >= 10:
            if flag == 0:
                mycard.start_hardware()
                mycard.vibrate_hardware(70)
                state = 'vibrating'
                print(state)
            else:
                mycard.start_hardware()
                mycard.vibrate_hardware(70)
                output = algo.law_1(first=oron)
                oron = False
                mycard.set_encoder_angle(output)
                # algo.plot_arrow(img)
                if algo.check_distance(5):
                    mycard.stop_hardware()
                    cv2.waitKey(5000)
                    print('arrive')
                    break
        else:
            flag = 1
            if first_time == 0:
                mycard.stop_hardware()
                cv2.waitKey(2000)
            state = 'point the motor to goal'
            print(state)

            if first_time == 0:
                mycard.calibrate()
                cv2.waitKey(3000)
                first_time = 1

            state = 'lets move to goal'
            print(state)
            if not algo.check_distance(5):
                algo.update(circle_center)
                mycard.start_hardware()
                output = algo.law_1(first=oron)
                print('output is is is is', output)
                delta_list.append(output)
                mycard.set_encoder_angle(output)
                mycard.vibrate_hardware(70)
                algo.plot_desired_position(img)
                algo.plot_path(img)
                # algo.plot_arrow(img)
                print('mamamama')
            else:
                mycard.stop_hardware()
                cv2.waitKey(5000)
                print('stop')
                break

    out.write(img)
    algo.display_image(img, circle_center, circle_radius)
    cv2.imshow('QueryImage', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Interrupted by user')
        break
    elif cv2.waitKey(1) & 0xFF == ord('i'):
        algo.position_user_input(img)

cam.release()
out.release()
cv2.destroyAllWindows()