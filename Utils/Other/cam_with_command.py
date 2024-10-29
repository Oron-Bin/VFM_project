import os
import datetime
import threading
import csv
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from Utils.Control.cardalgo import *
from Utils.Hardware.package import *

# Global constants
VIDEO_DIR = "/home/roblab20/Desktop/article_videos/full_algo"
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'XVID')

# Global variables
motor_angle_list = [0]


def command_listener(
        card, vibration_var, encoder_var, calibrate_btn_var, start_btn_var, stop_btn_var , vibration_var_2 , calibrate_btn_var_2, stop_btn_var_2, start_btn_var_2):
    """
    Thread function to listen for commands and control hardware accordingly.
    """
    hardware_started = False
    hardware_started_2 = False
    last_encoder_value = 0  # Track the last encoder value

    while True:
        if stop_btn_var.get() == 1:
            print("Stopping hardware...")
            card.stop_hardware()
            hardware_started = False
            vibration_var.set(0)
            encoder_var.set(0)
            stop_btn_var.set(0)
            last_encoder_value = 0  # Reset the last encoder value on stop
            print("Hardware stopped and sliders reset.")

        if calibrate_btn_var.get() == 1:
            print("Calibrating...")
            card.calibrate()
            calibrate_btn_var.set(0)
            last_encoder_value = 0  # Reset the last encoder value on calibration
            motor_angle_list.clear()  # Clear angle list
            motor_angle_list.append(0)  # Start angle list with zero after calibration
            print("Calibration done.")

        if start_btn_var.get() == 1:
            print("Starting hardware...")
            card.start_hardware()
            hardware_started = True
            start_btn_var.set(0)
            print("Hardware started.")

        if hardware_started:
            card.vibrate_hardware(vibration_var.get())
            current_encoder_value = encoder_var.get()
            encoder_difference = current_encoder_value - last_encoder_value
            card.set_encoder_angle(encoder_difference)
            motor_angle_list.append(encoder_difference + motor_angle_list[-1])
            last_encoder_value = current_encoder_value  # Update the last encoder value

        if stop_btn_var_2.get() == 1:
            print("Stopping hardware 2...")
            card.stop_hardware_2()
            hardware_started_2 = False
            vibration_var_2.set(0)
            encoder_var.set(0)
            stop_btn_var_2.set(0)
            last_encoder_value = 0  # Reset the last encoder value on stop
            print("Hardware stopped and sliders reset.")

        if calibrate_btn_var_2.get() == 1:
            print("Calibrating 2...")
            card.calibrate_2()
            calibrate_btn_var_2.set(0)
            last_encoder_value = 0  # Reset the last encoder value on calibration
            motor_angle_list.clear()  # Clear angle list
            motor_angle_list.append(0)  # Start angle list with zero after calibration
            print("Calibration done.")

        if start_btn_var_2.get() == 1:
            print("Starting hardware 2...")
            card.start_hardware_2()
            hardware_started_2 = True
            start_btn_var_2.set(0)
            print("Hardware started.")

        if hardware_started_2:
            card.vibrate_hardware_2(vibration_var_2.get())
            current_encoder_value = encoder_var.get()
            encoder_difference = current_encoder_value - last_encoder_value
            card.set_encoder_angle(encoder_difference)
            motor_angle_list.append(encoder_difference + motor_angle_list[-1])
            last_encoder_value = current_encoder_value  # Update the last encoder value

        time.sleep(0.1)


def main():
    """
    Main function to initialize the GUI and start the hardware control and video processing.
    """
    try:
        card = Card(x_d=0, y_d=0, a_d=-1, x=-1, y=-1, a=-1, baud=115200, port='/dev/ttyACM0')
        algo = card_algorithms(x_d=0, y_d=0)
        tip_pos = (300, 148)
        des_orientation = random.randint(0, 359)
        (algo.x_d, algo.y_d) = (290, 100)


    except Exception as e:
        print(f"Error initializing hardware: {e}")
        return
    root = tk.Tk()
    root.title("Hardware Control")
    vibration_var = tk.IntVar()
    encoder_var = tk.IntVar()
    calibrate_btn_var = tk.IntVar()
    start_btn_var = tk.IntVar()
    stop_btn_var = tk.IntVar()

    vibration_var_2 = tk.IntVar()
    calibrate_btn_var_2 = tk.IntVar()
    start_btn_var_2 = tk.IntVar()
    stop_btn_var_2 = tk.IntVar()

    def update_vibration_label(*args):
        vibration_label_var.set(f"Vibration: {vibration_var.get()}%")

    def update_vibration_label_2(*args):
        vibration_label_var_2.set(f"Vibration: {vibration_var_2.get()}%")

    def update_encoder_label(*args):
        encoder_label_var.set(f"Encoder: {encoder_var.get()}°")


    vibration_var.trace_add("write", update_vibration_label)
    vibration_var_2.trace_add("write", update_vibration_label_2)
    encoder_var.trace_add("write", update_encoder_label)

    vibration_label_var = tk.StringVar()
    vibration_label_var_2 = tk.StringVar()
    encoder_label_var = tk.StringVar()

    update_vibration_label()
    update_vibration_label_2()
    update_encoder_label()

    # Add widgets with appropriate row/column positions and spacing
    ttk.Label(root, text="Vibration (%)").grid(column=0, row=0, padx=10, pady=10)
    vibration_slider = ttk.Scale(root, from_=100, to=0, orient='vertical', variable=vibration_var)
    vibration_slider.grid(column=1, row=0, padx=10, pady=10)
    ttk.Label(root, textvariable=vibration_label_var).grid(column=2, row=0, padx=10, pady=10)

    ttk.Label(root, text="Vibration 2 (%)").grid(column=0, row=1, padx=10, pady=10)
    vibration_slider_2 = ttk.Scale(root, from_=100, to=0, orient='vertical', variable=vibration_var_2)
    vibration_slider_2.grid(column=1, row=1, padx=10, pady=10)
    ttk.Label(root, textvariable=vibration_label_var_2).grid(column=2, row=1, padx=10, pady=10)

    ttk.Label(root, text="Encoder (°)").grid(column=0, row=2, padx=10, pady=10)
    encoder_slider = ttk.Scale(root, from_=-180, to=180, orient='horizontal', variable=encoder_var)
    encoder_slider.grid(column=1, row=2, padx=10, pady=10)
    ttk.Label(root, textvariable=encoder_label_var).grid(column=2, row=2, padx=10, pady=10)

    # Place calibrate, start, and stop buttons with correct spacing
    # Place calibrate, start, and stop buttons in the same horizontal row
    calibrate_btn = ttk.Button(root, text="Calibrate", command=lambda: calibrate_btn_var.set(1))
    calibrate_btn.grid(column=0, row=3, padx=10, pady=10)

    start_btn = ttk.Button(root, text="Start", command=lambda: start_btn_var.set(1))
    start_btn.grid(column=1, row=3, padx=10, pady=10)

    stop_btn = ttk.Button(root, text="Stop", command=lambda: stop_btn_var.set(1))
    stop_btn.grid(column=2, row=3, padx=10, pady=10)

    # Place second set of calibrate, start, and stop buttons in the same horizontal row
    calibrate_btn_2 = ttk.Button(root, text="Calibrate 2", command=lambda: calibrate_btn_var_2.set(1))
    calibrate_btn_2.grid(column=0, row=4, padx=10, pady=10)

    start_btn_2 = ttk.Button(root, text="Start 2", command=lambda: start_btn_var_2.set(1))
    start_btn_2.grid(column=1, row=4, padx=10, pady=10)

    stop_btn_2 = ttk.Button(root, text="Stop 2", command=lambda: stop_btn_var_2.set(1))
    stop_btn_2.grid(column=2, row=4, padx=10, pady=10)

    def adjust_vibration(event):
        if event.keysym == "Up":
            vibration_var.set(min(vibration_var.get() + 1, 100))
        elif event.keysym == "Down":
            vibration_var.set(max(vibration_var.get() - 1, 0))

    def adjust_vibration_2(event):
        if event.keysym == "Up":
            vibration_var_2.set(min(vibration_var_2.get() + 1, 100))
        elif event.keysym == "Down":
            vibration_var_2.set(max(vibration_var_2.get() - 1, 0))

    def adjust_encoder(event):
        if event.keysym == "Right":
            encoder_var.set(min(encoder_var.get() + 1, 180))
        elif event.keysym == "Left":
            encoder_var.set(max(encoder_var.get() - 1, -180))

    root.bind("<Up>", adjust_vibration_2)
    root.bind("<Down>", adjust_vibration_2)
    root.bind("<Right>", adjust_encoder)
    root.bind("<Left>", adjust_encoder)

    command_thread = threading.Thread(target=command_listener, args=(
        card, vibration_var, encoder_var, calibrate_btn_var, start_btn_var, stop_btn_var , vibration_var_2 , calibrate_btn_var_2, stop_btn_var_2, start_btn_var_2))
    command_thread.daemon = True
    command_thread.start()

    cap = cv2.VideoCapture(0)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    video_path = os.path.join(VIDEO_DIR, f"output_{timestamp}.avi")
    out = cv2.VideoWriter(video_path, VIDEO_FOURCC, 20.0, (640, 480))
    start_time = time.time()
    angle_to_goal_list = [0]
    delta_angle_list = []
    final_list = [0]
    delta_final_list = []
    goal_list = [0]
    delta_target_list = []
    target_list = [0]

    CSV_FILE_PATH = "/home/roblab20/Desktop/article_videos/data_full_algo"
    csv_file_path = os.path.join(CSV_FILE_PATH, f"data_{timestamp}.csv")
    # Create CSV file and write headers
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Control angle', 'Orientation Angle', 'Radius', 'Time', 'Desire_Orientation','Orientation Error', 'R_Desire','Actual_R','Distance_Error','Phi_Desire' ,'Actual_Phi','Vibration Amp (%)'])
        # writer.writerow(['Control angle', 'Orientation Angle', 'Radius'])

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        elapsed_time = round(time.time() - start_time, 2)

        frame_copy = frame.copy()
        frame, centers = algo.detect_circles_and_get_centers(
            frame_copy)  # if not circle build anothe function for rectangles

        if len(centers) > 0:
            center = centers[0]
            algo.last_center = center
        else:
            center = algo.last_center


        if isinstance(center, tuple) and len(center) > 0:

            # cv2.circle(frame, tip_pos, radius=5, color=(0, 0, 0), thickness=1)

            cv2.circle(frame, (center[0], center[1]), 90, (0, 255, 0), 1)
            cv2.circle(frame, (algo.x_d, algo.y_d), radius=5, color=(0, 0, 255), thickness=2)

        print(f"Center: {center}")
        algo.path.extend(centers)
        aruco_centers, ids = algo.detect_aruco_centers(frame_copy)
        algo.arrow_coordinate_sys_motor(frame, tip_pos)
        # algo.plot_desired_position(frame)
        algo.plot_path(frame)

        if (aruco_centers and ids is not None):
            # for idx, center in enumerate(centers):
            print("aruco exists")
            print(center)
            if isinstance(center, tuple) and len(center) > 0:

                cv2.putText(frame, f"Center:{(center[0], center[1])}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                angle = algo.ids_to_angle(frame, ids, center, aruco_centers)  # the orientation angle
                if angle is not None:

                    orientation_error = abs(des_orientation - angle)
                    end_orientation = (round(center[0] + 50 * math.cos(np.deg2rad(angle))),
                                       round(center[1] + 50 * math.sin(np.deg2rad(angle))))

                    end = algo.rotate_point(center, end_orientation, 180)
                    end_des_orientation = (round(center[0] + 50 * math.cos(np.deg2rad(des_orientation))),
                                           round(center[1] + 50 * math.sin(np.deg2rad(des_orientation))))
                    end_des = algo.rotate_point(center, end_des_orientation, 180)

                    cv2.putText(frame, f"Orientation Angle: {angle}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    rotate_point = algo.rotate_point(tip_pos, (algo.x_d, algo.y_d), -180)
                    rotate_center = algo.rotate_point(tip_pos, center, 180)
                    command_angle = round(np.rad2deg(algo.find_dev(rotate_point, rotate_center)))
                    # command_angle = round(np.rad2deg(algo.find_dev(center, tip_pos)))
                    # control_angle = round(np.rad2deg(algo.find_dev(rotate_point, rotate_center)))
                    control_angle = round(np.rad2deg(algo.find_dev(tip_pos, rotate_center)))
                    rotate_control_angle = algo.rotate_point(((round(tip_pos[0] + 50 * math.cos(np.deg2rad(control_angle)))),
                                                             round(tip_pos[1] - 50 * math.sin(np.deg2rad(control_angle)))),center ,0)


                    # cv2.arrowedLine(frame, tuple(rotate_control_angle),tip_pos,
                                    # (255, 255, 0), 2)
                    cv2.arrowedLine(frame, center, tuple(end), (255, 255, 0), 2)
                    cv2.arrowedLine(frame, center, tuple(end_des), (255, 0, 0), 2)
                    cv2.putText(frame, f"control_angle: {control_angle}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    algo.plot_path(frame)
                    distance_to_goal = np.sqrt((center[0] - algo.x_d) ** 2 + (center[1] - algo.y_d) ** 2)
                    distance_to_tip = np.sqrt((center[0] - tip_pos[0]) ** 2 + (center[1] - tip_pos[1]) ** 2)
                    if not algo.orientation_achieved:
                        cv2.arrowedLine(frame, tuple(rotate_control_angle),tip_pos,
                        (0, 255, 0), 2)
                        if orientation_error < 5:
                            algo.flag = 1
                            print("Orientation error is less than 5 degrees")
                            stop_btn_var.set(1)
                            algo.stop_trigger = True
                            algo.orientation_achieved = True



                        else:
                            rotate_point = algo.rotate_point(tip_pos, (algo.x_d, algo.y_d), -180)
                            rotate_center = algo.rotate_point(tip_pos, center, 180)
                            command_angle = round(np.rad2deg(algo.find_dev(rotate_point, rotate_center)))
                            command_angle = round(np.rad2deg(algo.find_dev(center, tip_pos)))
                            # control_angle = round(np.rad2deg(algo.find_dev(rotate_point, rotate_center)))
                            control_angle = round(np.rad2deg(algo.find_dev(tip_pos, rotate_center)))
                            # tip_pos_2 = (338, 135)
                            # command_angle = round(np.rad2deg(algo.find_dev(center, tip_pos_2)))
                            angle_to_goal_list.append(command_angle)
                            print('want', command_angle)

                            delta_angle = angle_to_goal_list[-1] - angle_to_goal_list[-2]
                            delta_angle_list.append(delta_angle)
                            print(delta_angle_list[-1])

                            if len(delta_angle_list) <= 1:
                                vibration_var.set(80)
                                # encoder_var.set(50)
                                # card.set_encoder_angle(50)
                                encoder_var.set(command_angle)
                                card.set_encoder_angle(command_angle)
                            else:
                                print(distance_to_goal)

                                # encoder_var.set(1)
                                # card.set_encoder_angle(1)

                                if distance_to_goal < 5:
                                    print('enough')

                            print('orientation error is big')

                    if algo.orientation_achieved:
                        cv2.arrowedLine(frame, center ,(algo.x_d,algo.y_d),
                        (0, 0, 255), 1)
                        if algo.flag == 1:
                            # calibrate_btn_var_2.set(1)
                            algo.flag = 2
                            print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffff',algo.flag)

                        if distance_to_tip > 5 and algo.dis_to_tip_achieved == False :
                            # print(algo.dis_to_tip_achieved)
                            angle_to_tip = round(np.rad2deg(algo.find_dev(center, tip_pos)))
                            print('angle_to_tip', angle_to_tip)
                            final_list.append(angle_to_tip)
                            # print('want 2', angle_to_tip)

                            delta_final = final_list[-1] - final_list[-2]
                            delta_final_list.append(delta_final)

                            if len(delta_final_list) <= 1:
                                start_btn_var_2.set(1)

                                print('modifyyyyyyyyyyyyyyyyyyyyyyy')
                                # vibration_var.set(100)
                                # encoder_var.set(50)
                                # card.set_encoder_angle(50)
                                # if len(delta_final_list) > 0:
                                added_angle = delta_final_list[0] - angle_to_goal_list[1]

                                if added_angle < 0:
                                    added_angle +=360

                                else:
                                    added_angle = added_angle

                                encoder_var.set(added_angle)
                                card.set_encoder_angle(added_angle)
                                goal_list.append(added_angle)

                                print('dadadadadadadaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ',
                                      angle_to_goal_list[1] + added_angle)
                                # print('dadadadadadadaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ',
                                #       delta_final_list[0] )
                                # print('dadadadadadadaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ', delta_final_list[0] - angle_to_goal_list[1])


                            else:
                                vibration_var_2.set(100)
                                encoder_var.set(delta_final_list[-1])
                                card.set_encoder_angle(delta_final_list[-1])

                                print('distance to tip', distance_to_tip)

                        if distance_to_tip <= 5 and algo.dis_to_tip_achieved == False:
                            # vibration_var_2.set(0)
                            stop_btn_var_2.set(1)
                            algo.stop_trigger = True
                            print('enough_2')
                            algo.dis_to_tip_achieved = True

                        # elif distance_to_tip <= 5 and algo.dis_to_tip_achieved == True:
                        #     # vibration_var_2.set(0)
                        #     # stop_btn_var_2.set(1)
                        #     # algo.stop_trigger = True
                        #     print('enough_3')
                        #     # algo.dis_to_tip_achieved = True
                        #     # algo.flag = 3
                        #     # algo.flag = 3



                        if distance_to_goal > 5 and algo.dis_to_tip_achieved == True and algo.dis_to_goal_achieved == False and algo.go_to_goal == False:
                            # card.calibrate_2()
                            start_btn_var_2.set(1)
                            algo.go_to_goal = True

                        elif distance_to_goal >= 5 and algo.dis_to_tip_achieved == True and algo.dis_to_goal_achieved == False and algo.go_to_goal == True:

                            final_angle =  180 -(goal_list[1] + angle_to_goal_list[1]) - np.sum(np.diff(delta_final_list[1:]))

                            if final_angle < 0:
                                final_angle +=360
                            elif final_angle > 360:
                                final_angle -= 360

                            # encoder_var.set(final_angle)
                            # card.set_encoder_angle(final_angle)
                            print(final_angle,'ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
                            # print(angle_to_goal_list[1])
                            phi_desire = round(np.rad2deg(algo.find_dev((algo.x_d, algo.y_d), tip_pos)))
                            actual_phi = round(np.rad2deg(algo.find_dev((algo.x_d, algo.y_d), center)))
                            error_1 = phi_desire - actual_phi
                            target_list.append(error_1)
                            delta_target =target_list[-1] - target_list[-2]
                            delta_target_list.append(delta_target)
                            # algo.go_to_goal = True
                            if len(delta_target_list) <= 10:
                                calibrate_btn_var_2.set(1)
                                # encoder_var.set((int(input('please enter a number:'))))
                                # card.set_encoder_angle(int(input('please enter a number:')))
                                # start_btn_var_2.set(1)


                            else:
                                # print(delta_target_list)
                                # if algo.start_done == False:
                                #    start_btn_var_2.set(1)
                                #    algo.start_done = True
                                # else:
                                #     print('wating for youuuuuuuu_________________')
                                    # encoder_var.set(delta_target_list[-1])
                                    # card.set_encoder_angle(delta_target_list[-1])
                                # law_angle = algo.law_1(first= True)
                                # encoder_var.set(law_angle)
                                # card.set_encoder_angle(law_angle)

                                # encoder_var.set(delta_target_list[-1])
                                # card.set_encoder_angle(delta_target_list[-1])

                                # vibration_var_2.set(100)
                                print('wating for youuuuuuuu')
                                # vibration_var_2.set(100)
                                # algo.go_to_goal = True

                        # if distance_to_goal > 5 and algo.dis_to_tip_achieved == True and algo.go_to_goal == True:
                        #     vibration_var_2.set(100)
                        #     # print(delta_target_list)
                        #     # print(delta_target_list[-1])
                        #     # encoder_var.set(delta_target_list[-1])
                        #     # card.set_encoder_angle(delta_target_list[-1])
                        #     print('wating for youuuuuuuu')

                        # if distance_to_goal < 5 and algo.dis_to_tip_achieved == True and algo.go_to_goal == True:
                        if distance_to_goal < 5 and algo.dis_to_tip_achieved == True:
                            algo.dis_to_goal_achieved = True
                            stop_btn_var_2.set(1)
                            algo.stop_trigger = True
                        #
                        #     print('arrive with final orientation error of', orientation_error)
                        #     # start_btn_var.set(1)
                        #     start_point = tip_pos
                        #     end_point = (round(tip_pos[0] + 50 * math.cos(np.deg2rad(motor_angle_list[-1]))),
                        #                  round(tip_pos[1] - 50 * math.sin(np.deg2rad(motor_angle_list[-1]))))
                        #
                        #     rotated_end_point = algo.rotate_point(start_point, end_point, 90)
                        #     cv2.arrowedLine(frame, start_point, tuple(rotated_end_point), (0, 0, 255), 2)
                        #     print('waiting for order')

                    radius = 0.5 * (round(np.sqrt((center[0] - tip_pos[0]) ** 2 + (center[1] - tip_pos[1]) ** 2)))
                    goal_distance = 0.5 * (round(np.sqrt((algo.x_d - tip_pos[0]) ** 2 + (algo.y_d - tip_pos[1]) ** 2)))
                    phi_desire = round(np.rad2deg(algo.find_dev((algo.x_d,algo.y_d), tip_pos)))
                    actual_phi = round(np.rad2deg(algo.find_dev(center, tip_pos)))
                    phi_error = np.abs(phi_desire- actual_phi)
                    R_Error = np.abs(goal_distance -0.5*distance_to_tip )

                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        # 'Orientation Error', 'Distance Error', 'Vibration Amp (%)'
                        writer.writerow([control_angle, angle, radius, elapsed_time,des_orientation,orientation_error,goal_distance,0.5*distance_to_tip, R_Error,phi_desire,actual_phi,vibration_var_2.get()])

                else:
                    print("Failed to calculate orientation angle")
        else:

            print("aruco not exists")

        out.write(frame)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            root.quit()
            return

        root.after(10, update_frame)

    root.after(10, update_frame)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()