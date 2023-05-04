import cv2
import cv2.aruco as aruco
import numpy as np

# define ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

# define ArUco parameters
aruco_params = aruco.DetectorParameters_create()

# initialize camera
cap = cv2.VideoCapture(0)

# initialize variables for angle calculation
prev_marker_pos = None
curr_marker_pos = None

# loop over frames from the camera
while True:
    # read frame from camera
    ret, frame = cap.read()

    # detect ArUco markers in the frame
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    # if at least two markers detected
    if len(corners) >= 2:
        # draw marker borders
        aruco.drawDetectedMarkers(frame, corners, ids)

        # get position of first two markers
        curr_marker_pos = np.squeeze(corners[:2])

        # if this is not the first iteration
        if prev_marker_pos is not None:
            # calculate vector connecting first two markers
            vec = curr_marker_pos[1] - curr_marker_pos[0]

            # calculate angle of vector
            angle = np.arctan2(vec[1], vec[0]) * 180 / np.pi

            # print angle
            # print(f"Calculating angle between markers {ids[0]} and {ids[1]}")
            print('aa')
            print(f"Angle of vector connecting first two markers {ids[0]} and {ids[1]} is: {angle} degrees")


        # update previous marker position
        prev_marker_pos = curr_marker_pos

    # show the frame
    cv2.imshow('frame', frame)

    # check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close all windows
cap.release()
cv2.destroyAllWindows()
