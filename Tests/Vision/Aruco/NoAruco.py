import os
import sys
import datetime
import csv
import pandas as pd
import pickle

sys.path.insert(1, r'/')

from Utils.Control.cardalgo import *

initial_flag = 0

"""This Code is used to record a video"""
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"/home/roblab20/Desktop/videos/oron_videos/oron_{timestamp}.avi"
start_time = time.time()

# pickle_filename = f'/home/roblab20/Desktop/videos/data_oron/data_oron_{timestamp}.pickle'
csv_filename = f'/home/roblab20/Desktop/videos/data_oron/data_oron_{timestamp}.csv'
df = pd.DataFrame(columns=['Orientation','Pos_x','Pos_y','Motor_angle','delta_teta','Time'])

res = '720p'
data_list = []

def shortest_way(num_1, num_2):

    if abs(num_1 - num_2) < 180:
        return num_2 - num_1
    else:
        if num_1 > num_2:
            return abs(num_1 - num_2 - 360)
        else:
            return abs(num_1 - num_2) - 360

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

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current capture device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return VIDEO_TYPE[ext]
    # return VIDEO_TYPE['mp4']
    return VIDEO_TYPE['avi']


"""This part is responsible for the closed-loop control using CV2 circle detection"""
#==========================
#Defines camera parameters#
#==========================
cam = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 7, get_dims(cam, res))
cam.set(3,1280)
cam.set(4,720)

## Set the card class and open the serial communication
mycard = Card(x_d=0,y_d=0,a_d=-1,x=-1,y=-1,a=-1,baud=115200,port='/dev/ttyACM0')
mycard.set_motor_angle(0.0001) ## it was 0.0001 ## Update the motor angle value
mycard.send_data(key='motor') ## Send data to the motor/ this func in package
algo = card_algorithms(x_d=0,y_d=0) # Define the card algorithm object

# Define the set desired parameter and tell the code that the user didn't initialize it yet
set_des = 0
orientation_list = []
delta_list = []
flag = 0
j = 0

while cam.isOpened():
    ret, img = cam.read()
    time_diff = time.time() - start_time
    if ret:
        state_data = []
        circle_center, circle_radius = algo.detect_circle_info(img)
        aruco_centers, ids = algo.detect_aruco_centers(img)
        algo.finger_position(img,calibration=False) ## If Main axis system needs calibration change to True and calibrate the xy point
        print(aruco_centers)
        if set_des == 0: ## If the user didn't input a value yet
            print("No desired position yet")
            # algo.y_d = 227 ## 220
            # algo.x_d = 668

            algo.y_d = 227
            algo.x_d = 550
            start = time.perf_counter()
            print('the goal position is', algo.x_d,algo.y_d)

            set_des = 1

        elif circle_center is not None and set_des == 1: ## If the first card_center is not updated yet
            if (algo.card_initialize(circle_center)) == 1:
                set_des = 2
                mycard.vibrate_on()

        elif circle_center is not None:
            algo.plot_desired_position(img)
            algo.update(circle_center)
            algo.plot_path(img)

            if ids is not None and len(ids) > 0:
                orientation_angle = algo.ids_to_angle(ids, circle_center, aruco_centers)
                orientation_list.append(round(orientation_angle,1))
                if orientation_list and delta_list:
                    df = df.append({'Orientation': orientation_list[-1],
                                    'Pos_x': algo.path[-1][0],
                                    'Pos_y': algo.path[-1][-1],
                                    'Motor_angle': algo.angle_of_motor(),
                                    'delta_teta': delta_list[-1],
                                    'Time': time_diff}, ignore_index=True)

                # Draw arrowed line indicating orientation
                cv2.putText(img, f"Angle: {round(orientation_angle,1)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"goal_position: {algo.x_d, algo.y_d}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"my_position: {algo.path[-1]}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(img, f"motor_angle: {algo.angle_of_motor()}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"Time: {time_diff}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            origin = tuple(algo.finger_position(img)) #green point on screen
            scale = 50
            # Define the endpoints of the X-axis and Y-axis relative to the origin
            x_axis_end = (origin[0] - int(scale), origin[1])
            y_axis_end = (origin[0], origin[1] - int(scale))

            # Draw coordinate system
            cv2.line(img, origin, x_axis_end, (0, 0, 0), 2)  # X-axis (red)
            cv2.line(img, origin, y_axis_end, (0, 0, 0), 2)

            # Draw coordinate system
            cv2.line(img, origin, x_axis_end, (0, 0, 0), 2)  # X-axis (red)
            cv2.line(img, origin, y_axis_end, (0, 0, 0), 2)  # Y-axis (green)255

            if algo.check_distance(epsilon=10) is not True and set_des == 2: #there is a problem
                output  = algo.law_1()
                delta_list.append(output)

                cv2.putText(img, f"delta_motor_angle: {output}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(img, f"motor_angle: {algo.angle_of_motor()}", (10, 150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ###############################################

                mycard.set_encoder_angle(output) ## Update the motor output
                algo.plot_arrow(img) ## Plot the direction of the motor
                mycard.send_data('encoder') ## Send the motor output to the hardware
                time.sleep(0.1)

            elif algo.check_distance(10) is True:

                # print('Arrived at the goal', algo.path[-1])
                for i in range(30):
                    mycard.send_data('vibrate')
                    set_des = 3

            if set_des == 3:
                time.sleep(0.1) # a delay of a second between each iteration

                for i in range(30):
                    mycard.send_data('st') # or set des 3 or this is to stop the vibration
                # mycard.send_data('st')
                # time.sleep(1)
                algo.next_iteration()
                j = j + 1
                algo.package_data()

                algo.clear()
                algo.random_input()
                print('the new goal is',algo.random_input())
                set_des = 2

        out.write(img)
        algo.display_image(img, circle_center, circle_radius)
        # cv2.imshow('QueryImage', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interrupted by user')
            break
        if cv2.waitKey(1) & 0xFF == ord('i'):
            algo.position_user_input(img)

    df.to_csv(csv_filename, index=False)
    # with open(pickle_filename, 'wb') as f:
    #     pickle.dump(df, f)
    #
    # print("Data saved as pickle file:", pickle_filename)

    data = pd.read_csv(csv_filename)
    pixel_factor = 1/2 # convert pixel to mm
    # pixel_factor = 1.0
    data.at[0, 'x_dot'] = 0
    data.at[0, 'y_dot'] = 0
    data.at[0, 'phi_dot'] = 0


    data.at[len(data['Orientation'] -1 ), 'phi_dot'] = 0
    data.at[len(data['Orientation'] -1 ), 'x_dot'] = 0
    data.at[len(data['Orientation'] -1 ), 'y_dot'] = 0

    # data.at[0, 'Motor'] = 0
    data['Pos_x'] = (data['Pos_x']-data['Pos_x'][0]) * pixel_factor
    data['Pos_y'] = (data['Pos_y']-data['Pos_y'][0]) * pixel_factor

    # data.at[-1, 'phi_dot'] = 0
    # data.at[-1, 'x_dot'] = 0
    # data.at[-1, 'y_dot'] = 0
    # data.at[0, 'Pos_x'] = 0
    # data.at[0, 'Pos_y'] = 0
    for i in range(1,len(data['Orientation']) -1):
        # sub_phi = shortest_way(data['Orientation'][i-1], data['Orientation'][i]) #to try this one
        sub_phi = shortest_way(data['Orientation'][i],data['Orientation'][i-1])
        sub_x = (data['Pos_x'][i] - data['Pos_x'][i - 1])
        sub_y = (data['Pos_y'][i] - data['Pos_y'][i - 1])
        sub_time = data['Time'][i] - data['Time'][i-1]
        char_phi = sub_phi/sub_time
        char_x = sub_x / sub_time
        char_y = sub_y / sub_time
        data.at[i, 'phi_dot'] = char_phi
        data.at[i, 'x_dot'] = char_x
        data.at[i, 'y_dot'] = char_y
        data.at[i,'delta_teta'] = data.at[i+1, 'delta_teta']



    data.to_csv(csv_filename, index=False)



cam.release()
out.release()
cv2.destroyAllWindows()

