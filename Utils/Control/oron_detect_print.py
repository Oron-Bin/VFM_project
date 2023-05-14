# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy as np
import cv2
import cv2.aruco as aruco
import math
# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_250) # Every aruco marker has its own dictionary.

# rvec, tvec, _ = aruco.estimatePoseSingleMarkers(marker_corners, markerLength, camera_matrix, dist_coeffs)

def find_center_of_marker(marker_corners): #find the center of the marker
    x_sum = marker_corners[0][0][0] + marker_corners[0][1][0] + marker_corners[0][2][0] + marker_corners[0][3][0]
    y_sum = marker_corners[0][0][1] + marker_corners[0][1][1] + marker_corners[0][2][1] + marker_corners[0][3][1]
    x_center = x_sum * .25
    y_center = y_sum * .25
    return [x_center,y_center]

def find_dev(q1,q2): #return the angle between 2 points - oron add this to this code
    try:
        y = q2[1] - q1[1]
        x = q2[0] - q1[0]
        return math.degrees(math.atan2(y, x))
    except Exception as e:
        print("Error occured is find_dev error is:",e)

        # def mark_Aruco(img, marker_dict): #return the angle between 47 and 44
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     key_list = marker_dict.keys()
        #     for key in marker_dict.keys():
        #
        #         corner_1 = marker_dict.get(47)
        #         corner_2 = marker_dict.get(44)
        #         print(find_dev(corner_1,corner_2))

cam = cv2.VideoCapture(0) ## Create video camera object

while cam.isOpened(): ## When canmera port is open
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read() # Get image
    if ret: #boolian condition
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY) ## Covert to gray scale [0-255]

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS) ## Detect the markers and return their corners (x,y) relative to the camera axis system
        # return ids
        # print(corners)
        # Make sure all 5 markers were detected before printing them out
        if ids is not None:
            # Print corners and ids to the console
            key1= find_center_of_marker(corners[0]) #oron did
            key2=find_center_of_marker(corners[1]) #oron did
            # print (key1) #oron did

            # print(key2) #oron did
            print(ids)
            # corner_1 = ids.get(47)
            # corner_2 = ids.get(44)
            print(find_dev(key1, key2)) #oron did
            # print ('the orientation between',ids[0],'and',ids[1], 'is',find_dev(key1, key2)))


            for i, corner in zip(ids, corners):
                 print('ID: {}; Corners: {}'.format(i, corner))



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
