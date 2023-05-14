

import numpy as np
import cv2
import cv2.aruco as aruco
from math import * #when we write * it means to import the whole code
import matplotlib.pyplot as plt
import math
from matplotlib.path import Path
import matplotlib.patches as patches


def find_center_of_marker(marker_corners):
    x_sum = marker_corners[0][0][0] + marker_corners[0][1][0] + marker_corners[0][2][0] + marker_corners[0][3][0]
    y_sum = marker_corners[0][0][1] + marker_corners[0][1][1] + marker_corners[0][2][1] + marker_corners[0][3][1]
    x_center = x_sum * .25
    y_center = y_sum * .25
    # print([x_center,y_center])
    return (x_center,y_center)

def find_Aruco(QueryImg): ## This function returns dict
    # grayscale image
    gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_250)
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    if ids is not None:
        if ids.shape[0] > 1:
            blanket = {}  ## make a blanket dic for the ids and first corner
            for i, corner in zip(ids, corners):
                blanket[i[0]] = find_center_of_marker(corner)
            return blanket

def check_members(marker_dict): #check if the marker is in the dict -  oron mission - to fix this to general loop
    if 43 in marker_dict: ## and 47 in marker_dict
        return 1
    elif 44 in marker_dict:  ## and 47 in marker_dict
        return 2
    elif 45 in marker_dict:  ## and 47 in marker_dict
        return 3
    elif 46 in marker_dict: ## and 47 in marker_dict
        return 4
    else:
        return False


def find_dev(q1,q2): #return the angle between 2 points - this is a function that we build
    try:
        y = q2[1] - q1[1]
        x = q2[0] - q1[0]
        return math.degrees(math.atan2(y, x))
    except Exception as e:
        print("Error occured is find_dev error is:",e)



def mark_Aruco(img, marker_dict): #return the angle between 47 and 44 - oron to fix to general [roblem with general arucos
    font = cv2.FONT_HERSHEY_SIMPLEX # just font
    key_list = marker_dict.keys()
    if check_members(key_list) == 2:
        corner_1 = marker_dict.get(47)
        corner_2 = marker_dict.get(44)
        print(find_dev(corner_1,corner_2))



cam = cv2.VideoCapture(0)

while cam.isOpened():
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        my_dict = find_Aruco(QueryImg)
        if my_dict is not None:
             mark_Aruco(QueryImg,my_dict) #find the angle between the origin(47) and the other markers


        # Make sure all 5 markers were detected before printing them out

        # Display our image
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
