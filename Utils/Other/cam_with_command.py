import cv2
import threading
import serial
import os
import time
import datetime
from Utils.Hardware.package import Card
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
    video_path = f"/home/roblab20/Desktop/exp/output_{timestamp}.avi"  # Change this to your desired directory path
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        out.write(frame)  # Write the frame to the video file

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()