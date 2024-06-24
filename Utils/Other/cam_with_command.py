import cv2
import threading
import serial
import os
import time
import datetime
from Utils.Hardware.package import Card
import numpy as np

def jsonize(key, data):
    packet = 'json:{"'+str(key)+'":'+str(data)+'}'+'\x0d'+'\x0a'
    return packet


def command_listener(card):
    while True:
        command = input("Enter command: ")
        if command == "calibrate":
            card.calibrate()
        elif command == "start":
            card.start_hardware()
        elif command == "stop":
            card.stop_hardware()
        elif command == "vibrate":
            try:
                percent = int(input("Enter vibration percentage: "))
                card.vibrate_hardware(percent)
            except ValueError:
                print("Invalid vibration percentage")
        elif command.startswith("encoder"):
            try:
                angle = int(input("Enter motor angle: "))
                card.set_encoder_angle(angle)
            except ValueError:
                print("Invalid encoder command")
        else:
            print("Unknown command")


def detect_circles_and_get_centers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 1000, minRadius=50, maxRadius=300)

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

    command_thread = threading.Thread(target=command_listener, args=(card,))
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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_copy = frame.copy()  # Create a copy of the frame for drawing purposes

        frame, centers = detect_circles_and_get_centers(frame_copy)

        # Display circle centers on the screen
        for idx, center in enumerate(centers):
            cv2.putText(frame, f"{(center[0],center[1])}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)  # Write the frame to the video file
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()