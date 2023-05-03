from math import *
import sys
sys.path.insert(1, r'/')
from Code.Utils.Control.cardalgo import *

# from simple_pid import PID
import serial
nucleo = serial.Serial(timeout=0.000001)
nucleo.port = 'COM6'
nucleo.baudrate = 115200
initial_flag = 0






def find_center_of_marker(marker_corners):
    x_sum = marker_corners[0][0][0] + marker_corners[0][1][0] + marker_corners[0][2][0] + marker_corners[0][3][0]
    y_sum = marker_corners[0][0][1] + marker_corners[0][1][1] + marker_corners[0][2][1] + marker_corners[0][3][1]
    x_center = x_sum * .25
    y_center = y_sum * .25
    # print([x_center,y_center])
    return (x_center,y_center)


def find_Aruco(QueryImg): ## This function returns
    # grayscale image
    gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_250)
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    if ids is not None:
        if ids.shape[0] > 1:
            blanket = {}  ## make a blanket list for the ids and first corner
            for i, corner in zip(ids, corners):
                blanket[i[0]] = find_center_of_marker(corner)
            return blanket

def mark_Aruco(img, marker_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX
    key_list = marker_dict.keys()
    if check_members(key_list) == 1:
        corner_1,corner_2 = invert(marker_dict,43,44)
        # draw_axis(img, main_axis)
        angle_1 = orientation(img, corner_1, corner_2, font)
        # print("Marker 43:{} Marker 44:{}, Main Axis:{}, Orientation:{}".format(corner_1,corner_2,main_axis,angle_1))
        card_center = find_center_of_card(img,corner_1,corner_2)

    elif check_members(key_list) == 2:
        corner_3, corner_4 = invert(marker_dict, 45, 46)
        # draw_axis(img,main_axis)
        angle_2 = orientation(img, corner_3, corner_4, font)
        # print("Marker 45:{} Marker 46:{}, Main Axis:{}, Orientation:{}".format(corner_3,corner_4,main_axis,angle_2))
        card_center = find_center_of_card(img,corner_4,corner_3)
    else:
        # card_center = [0,0]
        return [img,None]
        # cv2.line(img, tuple(corner_2), tuple(corner_3), (255, 0, 0), 4)
        # cv2.line(img, tuple(corner_3), tuple(corner_4), (255, 0, 0), 4)
        # cv2.line(img, tuple(corner_4), tuple(corner_1), (255, 0, 0), 4)
    for key in key_list:
        COM = marker_dict[key]
        cv2.putText(img, str(key), (int(COM[0]+20), int(COM[1])), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return [img,card_center]



def draw_axis(img,point):
    cv2.line(img, tuple(point), (point[0]+20,point[1]), (150, 100, 255), 4)
    cv2.line(img, tuple(point), (point[0], point[1] -20 ), (255, 100, 200), 4)


def check_members(marker_dict):
    if 43 in marker_dict and 44 in marker_dict: ## and 47 in marker_dict
        return 1
    if 45 in marker_dict and 46 in marker_dict: ## and 47 in marker_dict
        return 2
    else:
        return False

def find_dev(q1,q2):
    y = q2[1] - q1[1]
    x = q2[0] - q1[0]
    return math.atan2(y,x)

def Map(inval):
    if 0 < inval < math.pi:
        inval = ((inval-0)*(180-0))/(math.pi-0) + 0
    elif -math.pi < inval < 0:
        inval = ((inval+math.pi)*(360-180))/(0+math.pi) + 180
    return inval

def invert(dict,num1,num2):
    corner_1 = []
    corner_2 = []
    main_axis = []
    for i in range(2):
        corner_1.append(int(dict[num1][i]))
        corner_2.append(int(dict[num2][i]))
        # main_axis.append(int(dict[47][i]))
    return corner_1,corner_2 ## , main_axis

def orientation(img,corner_1,corner_2,font):
    teta_degree = -1 * find_dev(corner_1, corner_2)
    teta = Map(float(teta_degree))
    # cv2.line(img, tuple(corner_1), tuple(corner_2), (255, 0, 0), 4)  # marking the centre of aruco
    # cv2.line(img, tuple(corner_1), (corner_1[0] + 400, corner_1[1]), (0, 0, 255), 4)  # marking the centre of aruco
    cv2.putText(img, "{0:.2f}".format(teta), (corner_1[0] + 75, corner_1[1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return teta

def find_center_of_card(img,corner_1,corner_2):
    A2_1 = np.array([[1,0,0,105],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) ## [[1,0,0,125],[0,1,0,125],[0,0,1,0],[0,0,0,1]] rec
    teta_degree =  find_dev(corner_1, corner_2)
    # teta_degree = np.arctan2(corner_1[0],corner_1[1])
    teta = radians(Map(float(teta_degree)))
    xval = corner_1[0]
    yval = corner_1[1]
    s = np.sin(teta)
    c = np.cos(teta)
    A1_0 = np.array([[c,-s, 0, xval],[s,c,0, yval],[0,0,1,0],[0,0,0,1]])
    A2_0 = A1_0.dot(A2_1)
    x_center = round(A2_0[0][3])
    y_center = round(A2_0[1][3])
    cv2.circle(img, (x_center, y_center), radius=5, color=(0, 0, 255), thickness=3)
    print("Center of card X: {} Y: {}".format(x_center,y_center))
    return (x_center,y_center)

def finger_position(img): ## Finger position is [317.0,171.0]
    position = (622,323)
    cv2.circle(img, position, radius=5, color=(0, 255, 0), thickness=3)
    return  position

## Card dimension is 86*54 mm
## marker dimension is 11mm
## for calculating center of the card we need to use 2 marker for finding orientation

def desired_location(img,x,y):
    cv2.circle(img, (x, y), radius=5, color=(255,0, 0), thickness=3)
    return


def EF(prev,new,weight):
    output = weight*new + (1-weight)*prev
    return output

def position_user_input():
    print('Enter desired X location')
    desired_x = int(input())
    print('Enter desired Y location')
    desired_y = int(input())
    print("Desired X is:{} Desired Y is:{}".format(desired_x, desired_y))
    return desired_x, desired_y


des_x,des_y = position_user_input()


