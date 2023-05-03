# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_250)

def find_center_of_marker(marker_corners):
    x_sum = marker_corners[0][0][0] + marker_corners[0][1][0] + marker_corners[0][2][0] + marker_corners[0][3][0]
    y_sum = marker_corners[0][0][1] + marker_corners[0][1][1] + marker_corners[0][2][1] + marker_corners[0][3][1]
    x_center = x_sum * .25
    y_center = y_sum * .25
    return [x_center,y_center]

cam = cv2.VideoCapture(0)

while cam.isOpened():
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        # print(corners)
        # Make sure all 5 markers were detected before printing them out
        if ids is not None:
            # Print corners and ids to the console
            print(find_center_of_marker(corners[0]))
            # for i, corner in zip(ids, corners):
            #     print('ID: {}; Corners: {}'.format(i, corner))

            # Outline all of the markers detected in our image
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

            # Wait on this frame
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # Display our image
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()