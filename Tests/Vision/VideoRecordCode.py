import cv2
import datetime

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"/home/roblab20/Desktop/videos/output_{timestamp}.avi"
out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

# Start recording
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Write the captured frame to the VideoWriter object
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close the display window
cv2.destroyAllWindows()


# import os
# import cv2
#
#
# cap = cv2.VideoCapture(0)
# out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))
#
# while True:
#     ret, frame = cap.read()
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()