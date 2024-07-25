import os
import cv2
import math
import time
import random
import datetime
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk

from Utils.Control.cardalgo import *
from Utils.Hardware.package import *

# Global constants
VIDEO_DIR = "/home/roblab20/Desktop/gui_exp"
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'XVID')

# Global variables
motor_angle_list = [0]



def command_listener(card, vibration_var, encoder_var, calibrate_btn_var, start_btn_var, stop_btn_var):
    """
    Thread function to listen for commands and control hardware accordingly.
    """
    hardware_started = False
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


        time.sleep(0.1)


def main():
    """
    Main function to initialize the GUI and start the hardware control and video processing.
    """
    try:
        card = Card(x_d=0, y_d=0, a_d=-1, x=-1, y=-1, a=-1, baud=115200, port='/dev/ttyACM0')
        algo = card_algorithms(x_d=0, y_d=0)
    except Exception as e:
        print(f"Error initializing hardware: {e}")
        return

    # tip_pos = (345, 120)
    tip_pos = (345, 150)
    des_orientation = random.randint(0, 359)
    # (algo.x_d, algo.y_d) = (370, 120)
    (algo.x_d, algo.y_d) = (350, 150)
    # print('now the angle to the goal from the tip is', round(np.rad2deg(np.arctan2(algo.y_d-120,algo.x_d - 345))))
    root = tk.Tk()
    root.title("Hardware Control")

    vibration_var = tk.IntVar()
    encoder_var = tk.IntVar()
    calibrate_btn_var = tk.IntVar()
    start_btn_var = tk.IntVar()
    stop_btn_var = tk.IntVar()


    # algo.orientation_achieved = False

    def update_vibration_label(*args):
        vibration_label_var.set(f"Vibration: {vibration_var.get()}%")

    def update_encoder_label(*args):
        encoder_label_var.set(f"Encoder: {encoder_var.get()}°")

    vibration_var.trace_add("write", update_vibration_label)
    encoder_var.trace_add("write", update_encoder_label)

    vibration_label_var = tk.StringVar()
    encoder_label_var = tk.StringVar()

    update_vibration_label()
    update_encoder_label()

    ttk.Label(root, text="Vibration (%)").grid(column=0, row=0, padx=10, pady=10)
    vibration_slider = ttk.Scale(root, from_=100, to=0, orient='vertical', variable=vibration_var)
    vibration_slider.grid(column=1, row=0, padx=10, pady=10)
    ttk.Label(root, textvariable=vibration_label_var).grid(column=2, row=0, padx=10, pady=10)

    ttk.Label(root, text="Encoder (°)").grid(column=0, row=1, padx=10, pady=10)
    encoder_slider = ttk.Scale(root, from_=-180, to=180, orient='horizontal', variable=encoder_var)
    encoder_slider.grid(column=1, row=1, padx=10, pady=10)
    ttk.Label(root, textvariable=encoder_label_var).grid(column=2, row=1, padx=10, pady=10)

    calibrate_btn = ttk.Button(root, text="Calibrate", command=lambda: calibrate_btn_var.set(1))
    calibrate_btn.grid(column=0, row=2, columnspan=2, padx=10, pady=10)

    start_btn = ttk.Button(root, text="Start", command=lambda: start_btn_var.set(1))
    start_btn.grid(column=0, row=3, columnspan=2, padx=10, pady=10)

    stop_btn = ttk.Button(root, text="Stop", command=lambda: stop_btn_var.set(1))
    stop_btn.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

    def adjust_vibration(event):
        if event.keysym == "Up":
            vibration_var.set(min(vibration_var.get() + 1, 100))
        elif event.keysym == "Down":
            vibration_var.set(max(vibration_var.get() - 1, 0))

    def adjust_encoder(event):
        if event.keysym == "Right":
            encoder_var.set(min(encoder_var.get() + 1, 180))
        elif event.keysym == "Left":
            encoder_var.set(max(encoder_var.get() - 1, -180))

    root.bind("<Up>", adjust_vibration)
    root.bind("<Down>", adjust_vibration)
    root.bind("<Right>", adjust_encoder)
    root.bind("<Left>", adjust_encoder)

    command_thread = threading.Thread(target=command_listener, args=(
        card, vibration_var, encoder_var, calibrate_btn_var, start_btn_var, stop_btn_var))
    command_thread.daemon = True
    command_thread.start()

    cap = cv2.VideoCapture(0)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    video_path = os.path.join(VIDEO_DIR, f"output_{timestamp}.avi")
    out = cv2.VideoWriter(video_path, VIDEO_FOURCC, 20.0, (640, 480))

    angle_to_goal_list = [0]
    smooth_list = [0]
    delta_angle_list = []
    def update_frame():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        frame_copy = frame.copy()

        frame, centers = algo.detect_circles_and_get_centers(frame_copy) # if not circle build anothe function for rectangles
        algo.path.extend(centers)
        aruco_centers, ids = algo.detect_aruco_centers(frame_copy)
        algo.arrow_coordinate_sys_motor(frame, tip_pos)
        algo.plot_desired_position(frame)
        algo.plot_path(frame)

        if aruco_centers and ids is not None:
            for idx, center in enumerate(centers):

                print(center)
                cv2.putText(frame, f"{(center[0], center[1])}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                angle = algo.ids_to_angle(frame, ids, center, aruco_centers)
                if angle is not None:

                    orientation_error = abs(des_orientation - angle)
                    end_orientation = (round(center[0] + 50 * math.cos(np.deg2rad(angle))),
                                       round(center[1] + 50 * math.sin(np.deg2rad(angle))))
                    # print('end =',end_orientation)
                    # print(center)
                    end = algo.rotate_point(center,end_orientation,180)
                    # print('end =',end)
                    end_des_orientation = (round(center[0] + 50 * math.cos(np.deg2rad(des_orientation))),
                                           round(center[1] + 50 * math.sin(np.deg2rad(des_orientation))))
                    end_des = algo.rotate_point(center,end_des_orientation,180)

                    cv2.putText(frame, f"Orientation: {angle}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Orientation_error: {orientation_error}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    cv2.arrowedLine(frame, center, tuple(end), (255, 255, 0), 2)
                    cv2.arrowedLine(frame, center, tuple(end_des), (255, 0, 0), 2)

                    # print(rad_angle)
                    algo.plot_path(frame)
                    distance = np.sqrt((center[0] - algo.x_d) ** 2 + (center[1] - algo.y_d) ** 2)

                    if not algo.orientation_achieved:
                        if orientation_error < 5:
                            algo.flag = 1

                            print("Orientation error is less than 5 degrees")
                            stop_btn_var.set(1)
                            calibrate_btn_var.set(1)
                            cv2.waitKey(2000)
                            algo.stop_trigger = True
                            algo.orientation_achieved = True
                            algo.clear()
                        else:
                            # vibration_var.set(70)  # Set vibration to 60%
                            # card.vibrate_hardware(70)

                            # encoder_var.set()
                            # card.set_encoder_angle(50)
                            print('orientation error is big')

                    if algo.orientation_achieved:
                        start_btn_var.set(1)
                        if distance < 5:
                            # angle_to_goal_list = []
                            print('arrive with final orientation error of', orientation_error)
                            stop_btn_var.set(1)
                            algo.clear()
                            break
                        else:
                            # vibration_var.set(100)  # Set vibration to 60%
                            # card.vibrate_hardware(100)
                            # start_btn_var.set(1)
                            rotate_point = algo.rotate_point(tip_pos, (algo.x_d, algo.y_d), -180)
                            rotate_center = algo.rotate_point(tip_pos, center, 180)
                            rad_angle = round(np.rad2deg(algo.find_dev(rotate_point, rotate_center)))
                            angle_to_goal_list.append(rad_angle)
                            print('want', rad_angle)
                            smooth_angle = round(0.1 * angle_to_goal_list[-1] + 0.9 * angle_to_goal_list[-2])
                            # print(smooth_angle)
                            smooth_list.append(smooth_angle)
                            smooth_delta = smooth_list[-1] - smooth_list[-2]
                            delta_angle = angle_to_goal_list[-1] - angle_to_goal_list[-2]

                            delta_angle_list.append(delta_angle)
                            print(delta_angle_list[-1])

                            if len(delta_angle_list) <= 1:
                                encoder_var.set(delta_angle_list[-1])
                                card.set_encoder_angle(delta_angle_list[-1])
                                vibration_var.set(100)  # Set vibration to 60%
                                card.vibrate_hardware(100)
                            else:
                                encoder_var.set(delta_angle_list[-1])
                                card.set_encoder_angle(delta_angle_list[-1])
                                vibration_var.set(100)  # Set vibration to 60%
                                card.vibrate_hardware(100)

                            # print('The distance is big, angle to goal:', rad_angle)


                else:
                    print("Failed to calculate orientation angle")

        if len(motor_angle_list) > 0:
            start_point = tip_pos
            end_point = (round(tip_pos[0] + 50 * math.cos(np.deg2rad(motor_angle_list[-1]))),
                         round(tip_pos[1] - 50 * math.sin(np.deg2rad(motor_angle_list[-1]))))
            # print(motor_angle_list)
            rotated_end_point = algo.rotate_point(start_point, end_point, 90)
            cv2.arrowedLine(frame, start_point, tuple(rotated_end_point), (0, 0, 255), 2)

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
