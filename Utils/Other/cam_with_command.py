import cv2
import threading
import os
import time
import datetime
from Utils.Hardware.package import Card
import numpy as np
import math
import tkinter as tk
from tkinter import ttk


def jsonize(key, data):
    packet = 'json:{"' + str(key) + '":' + str(data) + '}' + '\x0d' + '\x0a'
    return packet


def rotate_point(center, point, angle):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    point_shifted = np.array(point) - np.array(center)
    rotated_point_shifted = rotation_matrix.dot(point_shifted)
    rotated_point = rotated_point_shifted + np.array(center)
    return rotated_point.astype(int)


angle_list = [0]


def command_listener(card, vibration_var, encoder_var, calibrate_btn_var, start_btn_var, stop_btn_var):
    hardware_started = False
    last_encoder_value = 0  # Track the last encoder value

    while True:
        if calibrate_btn_var.get() == 1:
            print("Calibrating...")
            card.calibrate()
            calibrate_btn_var.set(0)
            last_encoder_value = 0  # Reset the last encoder value on calibration
            print("Calibration done.")

        if start_btn_var.get() == 1:
            print("Starting hardware...")
            card.start_hardware()
            hardware_started = True
            start_btn_var.set(0)
            print("Hardware started.")

        if stop_btn_var.get() == 1:
            print("Stopping hardware...")
            card.stop_hardware()
            hardware_started = False
            vibration_var.set(0)
            encoder_var.set(0)
            stop_btn_var.set(0)
            last_encoder_value = 0  # Reset the last encoder value on stop
            print("Hardware stopped and sliders reset.")

        if hardware_started:
            # Apply the vibration setting
            card.vibrate_hardware(vibration_var.get())

            # Calculate the difference for the encoder
            current_encoder_value = encoder_var.get()
            encoder_difference = current_encoder_value - last_encoder_value
            card.set_encoder_angle(encoder_difference)
            angle_list.append(encoder_difference + angle_list[-1])
            last_encoder_value = current_encoder_value  # Update the last encoder value

        time.sleep(0.1)


tip_pos = (323, 150)


def detect_circles_and_get_centers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 1000, minRadius=50, maxRadius=300)
    cv2.circle(frame, tip_pos, radius=5, color=(0, 0, 0), thickness=2)

    centers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            centers.append((x, y))
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), radius=5, color=(255, 255, 0), thickness=2)
    return frame, centers


def main():
    card = Card(x_d=0, y_d=0, a_d=-1, x=-1, y=-1, a=-1, baud=115200, port='/dev/ttyACM0')

    root = tk.Tk()
    root.title("Hardware Control")

    vibration_var = tk.IntVar()
    encoder_var = tk.IntVar()
    calibrate_btn_var = tk.IntVar()
    start_btn_var = tk.IntVar()
    stop_btn_var = tk.IntVar()

    ttk.Label(root, text="Vibration (%)").grid(column=0, row=0, padx=10, pady=10)
    vibration_slider = ttk.Scale(root, from_=0, to=100, orient='horizontal', variable=vibration_var)
    vibration_slider.grid(column=1, row=0, padx=10, pady=10)

    ttk.Label(root, text="Encoder (°)").grid(column=0, row=1, padx=10, pady=10)
    encoder_slider = ttk.Scale(root, from_=0, to=360, orient='horizontal', variable=encoder_var)
    encoder_slider.grid(column=1, row=1, padx=10, pady=10)

    calibrate_btn = ttk.Button(root, text="Calibrate", command=lambda: calibrate_btn_var.set(1))
    calibrate_btn.grid(column=0, row=2, columnspan=2, padx=10, pady=10)

    start_btn = ttk.Button(root, text="Start", command=lambda: start_btn_var.set(1))
    start_btn.grid(column=0, row=3, columnspan=2, padx=10, pady=10)

    stop_btn = ttk.Button(root, text="Stop", command=lambda: stop_btn_var.set(1))
    stop_btn.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

    command_thread = threading.Thread(target=command_listener, args=(
    card, vibration_var, encoder_var, calibrate_btn_var, start_btn_var, stop_btn_var))
    command_thread.daemon = True
    command_thread.start()

    cap = cv2.VideoCapture(0)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Specify the directory path for saving the video
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    video_path = f"/home/roblab20/Desktop/experiments/output_{timestamp}.avi"
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    circle_centers = []

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        frame_copy = frame.copy()  # Create a copy of the frame for drawing purposes

        frame, centers = detect_circles_and_get_centers(frame_copy)

        if len(angle_list) > 1:
            start_point = tip_pos
            end_point = (round(tip_pos[0] + 50 * math.cos(np.deg2rad(angle_list[-1]))),
                         round(tip_pos[1] - 50 * math.sin(np.deg2rad(angle_list[-1]))))
            rotated_end_point = rotate_point(start_point, end_point, 90)

            # Draw the rotated arrow
            cv2.arrowedLine(frame, start_point, tuple(rotated_end_point), (0, 0, 255), 2)

        # Display circle centers on the screen
        for idx, center in enumerate(centers):
            cv2.putText(frame, f"{(center[0], center[1])}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)  # Write the frame to the video file
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            root.quit()
            return

        root.after(10, update_frame)  # Schedule the next update

    root.after(10, update_frame)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
