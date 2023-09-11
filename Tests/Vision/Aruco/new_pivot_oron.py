import os
import sys
import datetime

from Utils.Control.cardalgo import *

sys.path.insert(1, r'/')
initial_flag = 0

"""This Code is used to record a video"""
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"/home/roblab20/Desktop/videos/new_orientation/oron_{timestamp}.avi"
start_time = time.time()

res = '720p'

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

cam = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 7, get_dims(cam, res))
cam.set(3,1280)
cam.set(4,720)

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
        # state_data = []
        circle_center, circle_radius = algo.detect_circle_info(img)
        aruco_centers, ids = algo.detect_aruco_centers(img)
        origin = tuple(algo.finger_position(img))
        algo.finger_position(img,calibration=False) ## If Main axis system needs calibration change to True and calibrate the xy point
        # print(aruco_centers)
        if set_des == 0: ## If the user didn't input a value yet
            print("No desired position yet")
            algo.y_d = origin[1] ## 220
            algo.x_d = origin[0]

            algo.orientation = random.randint(0,359)

            start = time.perf_counter()
            print('the goal pose is', algo.x_d,algo.y_d, algo.orientation)

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

                # Draw arrowed line indicating orientation
                cv2.putText(img, f"Angle: {round(orientation_angle,1)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"goal_position: {algo.x_d, algo.y_d}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"my_position: {algo.path[-1]}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # cv2.putText(img, f"motor_angle: {algo.angle_of_motor()}", (10, 150),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"desire_angle: {round(algo.orientation,0)}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(img, f"Time: {time_diff}", (10, 180),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"angle_error: {abs(orientation_angle - algo.orientation)}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            scale = 50
            # Define the endpoints of the X-axis and Y-axis relative to the origin
            x_axis_end = (origin[0] + int(scale), origin[1])
            y_axis_end = (origin[0], origin[1] + int(scale))

            # Draw coordinate system
            cv2.line(img, origin, x_axis_end, (0, 0, 0), 2)  # X-axis (red)
            cv2.line(img, origin, y_axis_end, (0, 0, 0), 2)

            # Draw coordinate system
            cv2.line(img, origin, x_axis_end, (0, 0, 0), 2)  # X-axis (red)
            cv2.line(img, origin, y_axis_end, (0, 0, 0), 2)  # Y-axis (green)255


            if algo.check_distance(epsilon=1) is not True and set_des == 2 :
                print('des_orientation', algo.orientation)
                print('current orientation', orientation_angle)
                orientation_error = abs(shortest_way(orientation_angle,algo.orientation))
                # orientation_error = abs(orientation_angle - algo.orientation)
                print('the error is',orientation_error)
                if orientation_error > 15 and flag == 0:
                    # flag = 0
                    print('condition reached and the error is', orientation_error)
                    print('**********************************************************')
                    print('**********************************************************')
                    print('**********************************************************')
                    print('**********************************************************')
                    print('**********************************************************')
                    print('**********************************************************')
                    # if flag == 0:

                    # output  = algo.law_1()
                    output = 0
                    print('output is',output)
                    cv2.putText(img, f"delta_motor_angle: {output}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    mycard.set_encoder_angle(output) ## Update the motor output
                    algo.plot_arrow(img) ## Plot the direction of the motor
                    mycard.send_data('encoder') ## Send the motor output to the hardware
                    time.sleep(0.001)
                elif orientation_error > 15 and flag ==1:
                    print('just ignore the error now')
                    output = algo.law_1()
                    print('output is', output)
                    mycard.set_encoder_angle(output)  ## Update the motor output
                    algo.plot_arrow(img)  ## Plot the direction of the motor
                    mycard.send_data('encoder')  ## Send the motor output to the hardware
                    time.sleep(0.001)

                    if algo.check_distance(10) is True:
                        print('flag number 1')
                        # # elif algo.check_distance(10) is True and (orientation_angle-des_orientation) <= 2 :
                        print('Arrived at the goal position', algo.path[-1], 'and orientation')
                        for i in range(30):
                            mycard.send_data('vibrate')
                        # print('Arrived at the goal pose !!!!')
                            time.sleep(3)
                            set_des = 3

                        if set_des == 3:
                            time.sleep(3)
                elif orientation_error <= 15:
                    # mycard.send_data('vibrate')
                    # time.sleep(2)
                    flag = 1 #because i dont want to fix the orientation error again
            # # elif algo.check_distance(epsilon=10) is not True and set_des == 2 and (orientation_angle-des_orientation) <= 2:
                    print('starting navigate to the point')

                    output = algo.law_1()
                    print('output is', output)
                    mycard.set_encoder_angle(output) ## Update the motor output
                    algo.plot_arrow(img) ## Plot the direction of the motor
                    mycard.send_data('encoder') ## Send the motor output to the hardware
                    time.sleep(0.001)
                    # mycard.send_data('st')
                    # time.sleep(0.001)

                    if algo.check_distance(10) is True:
                        # for i in range(30):
                        mycard.send_data('vibrate')
                        time.sleep(3)
                        print('flag number 2')
                # # elif algo.check_distance(10) is True and (orientation_angle-des_orientation) <= 2 :
                        print('Arrived at the goal pose finalllllllllllllll and the position')

                        # mycard.send_data('vibrate')
                        # time.sleep(3)
                        set_des = 3

                    if set_des == 3:
                        time.sleep(3)

        out.write(img)
        algo.display_image(img, circle_center, circle_radius)
        # cv2.imshow('QueryImage', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interrupted by user')
            break
        if cv2.waitKey(1) & 0xFF == ord('i'):
            algo.position_user_input(img)


cam.release()
out.release()
cv2.destroyAllWindows()