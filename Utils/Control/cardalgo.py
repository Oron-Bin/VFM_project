import cv2
import numpy as np
import random
import pandas as pd
from Utils.Hardware.package import *
import pickle
import cv2.aruco as aruco
import math
# from Utils.Control.robotiqGripper import *


## Card algortihm function schematics

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class card_algorithms:
    def __init__(self,x_d,y_d):
        self.x_d = x_d ## Card desired x value - int
        self.y_d = y_d ## Card desitred y value - int
        self.center = None ## Card center - list after using update function
        self.orientation = 0
        self.last_orientation = None
        self.first_orientation = None
        self.last_dx = None ## last_measurment
        self.last_dy = None
        self.last_angle = 0
        self.angle = None ##
        self.path = [] ## list of tuples which contains the card path
        self.orientation_list = []
        self.last_delta = None
        self.tip_position = None ## The position of the finger tip
        self.output = None ## Motor Angle output after algo
        self.error = None
        self.iteration = 0
        self._start_time = None
        self.elapsed_time = None
        self.time_list = []
        self.data  = pd.DataFrame(columns = ['path','orientation', 'desired_x','desired_y','iteration','time','error']) ## ,dtype='object'
        self.markers = None
        self.rec_path = None


    def update(self,center):

        """Update card and path position and time of position"""
        self.time_list.append(time.perf_counter()) ## Update the time
        # self.center = center ## update the center
        # self.path.append(center) ## update the path
        self.center = center ## update the center
        self.path.append(center) ## update the path
        self.orientation_list.append(self.orientation) ## update the orientation


    def clear(self):
        self.path.clear() ## Clear path list
        self.time_list.clear() ## Clear time list
        self.orientation_list.clear() ## Clear Orientation list

    def card_initialize(self,card_center): ## Use to intialize position for the first iteration

        """ function for initialize card data after starting the system"""

        self.center = card_center
        self.last_dx = self.x_d - self.center[0]
        self.last_dy = self.y_d - self.center[1]

        self.last_orientation = self.orientation
        self.first_orientation = self.orientation
        print("card_initialize")
        return 1

#        if (self.first_orientation != 0):
#             print("First orientation is {}:", self.first_orientation)
#             return 1
#         else:
#             return 0

    def plot_desired_position(self,img):

        """ plot a circle point in the desired position"""
        cv2.circle(img, (round(self.x_d), round(self.y_d)), radius=5, color=(255, 0, 0), thickness=3)

    def filter(self,prev,new,weight=0.5):

        """ exponent filter"""

        output = weight*new +(1-weight*prev)
        return output

    def law_1(self):

        """ Function for calculating the first control loop law"""
        dx = self.filter(self.last_dx,self.x_d - self.center[0])
        dy = self.filter(self.last_dy,self.y_d - self.center[1])


        # print(self.last_dx,self.x_d,self.center[0])
        # print('dx',dx)
        # print('dy',dy)
        new_angle = round(np.degrees(np.arctan2(dx,dy)))
        # print(new_angle)
        # print("New :{} Last is:{}".format(new_angle, self.last_angle))
        self.last_dx = dx
        self.last_dy = dy
        # print(self.last_dx)
        if self.last_angle == new_angle:
            # print('oooo')
            return 0
        else:
            # print('aaa')
            out = round(self.shortest_motor_path(new_angle))
            self.last_angle = new_angle
            return out


    def finger_position(self,img,calibration=False):  ## Finger position is [(632,256)]

        """ return the finger position after calibration"""

        if calibration == True:
            self.tip_position = self.point_calibration(img) #return the x,y of the function below
        else:
            self.tip_position = (644,185) #the center of the top
            cv2.circle(img, self.tip_position, radius=5, color=(0, 255, 0), thickness=3)
        return self.tip_position

    def point_calibration(self,img):
        """ Function for calibration the tip point"""

        print("Enter X value")
        x = int(input())
        print("Enter Y value")
        y = int(input())
        while True:
            cv2.circle(img, (x,y), radius=5, color=(0, 255, 0), thickness=3)
            cv2.imshow('QueryImage', img)

            print("Finish Calibration Enter y if Yes n to continue")
            status = str(input())
            if status == 'y':
                break
            else:
                print("Enter New X value")
                x = int(input())
                print("Enter New Y value")
                y = int(input())
                continue
        return (x,y)

    def position_user_input(self):

        """ User destenation request"""

        print('Enter desired X location')
        self.x_d = self.tip_position[0] + int(input())
        print('Enter desired Y location')
        self.y_d = self.tip_position[1] + int(input())
        # print('Enter desired orientation')
        # self.orientation = int(input())
        return [self.x_d,self.y_d]

    def random_input(self):
        """ Random new input inside the rectangle area"""
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        # phi = random.randint(0,360)
        self.x_d = self.tip_position[0] + x
        self.y_d = self.tip_position[1] + y
        self.orientation = random.randint(0, 359)
        print('Enter desired point is',self.x_d, self.y_d)
        return [self.x_d, self.y_d,self.orientation]

        # return [self.x_d, self.y_d, self.orientation]
    def generate_path(self):
        """ generate rectangle path"""
        x = np.linspace(-30,30,15) + self.tip_position[0]
        y = np.linspace(-30,30,15) + self.tip_position[1]

        path = []
        for i in x:
            point = [i, y[0]]
            path.append(point)
        for i in y:
            point = [x[-1], i]
            path.append(point)
        for i in reversed(x):
            point = [i, y[-1]]
            path.append(point)
        for i in reversed(y):
            point = [x[0], i]
            path.append(point)
        return path

    def generate_heart(self):
        t = np.arange(0, 2 * np.pi, 0.1)
        x = -30 * np.sin(t) ** 3 + self.tip_position[0]
        y = 30 * np.cos(t) - 10 * np.cos(2 * t) - 4 * np.cos(3 * t) - 2*np.cos(4 * t) + self.tip_position[1]
        heart = []
        for i in range(len(x)):
            heart.append([x[i],y[i]])
        h1 = heart[:round(len(x) / 2)]
        h2 = heart[round(len(x) / 2):]
        h2.extend(h1)
        return h2

    def generate_circle(self):
        t = np.arange(0, 2 * np.pi, 0.1)
        x = 30 * np.sin(t) + self.tip_position[0]
        y = 30 * np.cos(t) + self.tip_position[1]
        circle = []
        for i in range(len(x)):
            circle.append([x[i],y[i]])
        return list(reversed(circle))

    def plot_desired_path(self,img,start,end):
        color = (255,0,0)
        cv2.rectangle(img, start, end,color,1)

    def plot_path(self,img,thickness = 3, indices = 1):

        """ Plot the card total path every iteration"""
        # loop over the set of tracked points
        for i in range(1, len(self.path)):
            # if either of the tracked points are None, ignore
            # them
            if self.path[i - 1] is None or self.path[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            # thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(img, tuple(self.path[i - 1]), tuple(self.path[i]), (0, 0, 255), thickness)
        # return

    def plot_arrow(self,img):
        """ Plot Arrow of the Forcer Vector on 2D """

        cv2.arrowedLine(img, self.tip_position, (round(self.tip_position[0]+self.x_d-self.center[0]),round(self.tip_position[1]+self.y_d-self.center[1])),
                        (255,255,0), 2)

    def shortest_motor_path(self,output):

        """Function for finding the shortest motor path to destanation"""

        if abs(self.last_angle-output) < 180:
            return output - self.last_angle
        else:
            if self.last_angle>output:
                return abs(self.last_angle-output-360)
            else:
                return abs(self.last_angle-output)-360


    def output_calibrate(self):

        """ Manual control for motor angle calibration"""

        print("Input Angle")
        input_ = int(input())
        out = self.shortest_motor_path(input_)
        print("New :{} Last is:{}".format(input_, self.last_angle))
        self.last_angle = input_
        print(out)
        return out

    def detect_circle_info(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (8, 8))
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.5, 1000, minRadius=50, maxRadius=300)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                circle_center = [x, y]
                circle_radius = r

                return circle_center, circle_radius
        print("No circles detected")
        return None, None

    def display_image(self,image, circle_center, circle_radius):
        if circle_center is not None and circle_radius is not None:
            cv2.circle(image, tuple(circle_center), circle_radius, (0, 255, 0), 4)
            cv2.circle(image, tuple(circle_center), 3, (0, 0, 255), -1)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened_image = cv2.filter2D(image, -1, kernel)

        cv2.imshow('QueryImage', sharpened_image)
        cv2.waitKey(1)
    def detect_aruco_centers(self,frame):
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        aruco_params = cv2.aruco.DetectorParameters_create()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        corners, ids, rejected = cv2.aruco.detectMarkers(blurred, aruco_dict, parameters=aruco_params)

        aruco_centers = []
        if ids is not None:
            for i in range(len(ids)):
                aruco_center = np.mean(corners[i][0], axis=0)
                aruco_centers.append(aruco_center)
        # print(ids,aruco_centers)
        return aruco_centers, ids

    # def calculate_angle(self,circle_center, aruco_center):
    #     dx = circle_center[0] - aruco_center[0]
    #     dy = circle_center[1] - aruco_center[1]
    #     angle = np.degrees(np.arctan2(dy, dx))
    #
    #     return angle
    def calculate_angle(self,point1,point2):
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        angle = np.degrees(np.arctan2(dy, dx))

        return angle
    def ids_to_angle(self, ids, circle_center, aruco_centers):
        last_aruco_center = aruco_centers[-1]
        angle = self.calculate_angle(circle_center, last_aruco_center)
        if ids is not None:
            if ids[-1] == 43:
                # angle = calculate_angle(circle_center, last_aruco_center)
                angle = angle
                # cv2.arrowedLine(
                #     img,
                #     tuple(circle_center),
                #     tuple(last_aruco_center),
                #     (0, 0, 255),
                #     2,
                #     tipLength=0.2
                # )
            elif ids[-1] == 44:
                angle += 180
                # angle = calculate_angle(circle_center, last_aruco_center) + 180
            elif ids[-1] == 45:
                angle += 90
                # angle = calculate_angle(circle_center, last_aruco_center) +90
            elif ids[-1] == 46:
                angle -= 90
                # angle = calculate_angle(circle_center, last_aruco_center) -90
            if angle < 0:
                angle += 360
            # print('The orientation is:', angle)
            return angle

    def plot_angles(self,frame_numbers, angles):
        plt.plot(frame_numbers, angles)
        plt.xlabel('Frame')
        plt.ylabel('Angle')
        plt.ylim([0, 360])
        plt.title('Angle vs. Frame')
        plt.show()

    # def find_card_orientation(self, QueryImg):
    #
    #     """Finding the card orientation using Aruco markers"""
    #     #to check - if i need this function or if i need the 'orientaion' code
    #
    #     # grayscale image
    #     gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    #     ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_250)
    #     ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    #     # Detect Aruco markers
    #     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    #     if ids is not None:
    #         if ids.shape[0] > 1:
    #             blanket = {}  ## make a blanket list for the ids and first corner
    #             for i, corner in zip(ids, corners):
    #                 blanket[i[0]] = self.find_center_of_marker(corner)
    #             self.markers = blanket
    #             # print(self.markers)
    #             # if self.markers == None:
    #             #     self.find_orientation(QueryImg)
    #             new_orientation = self.find_orientation(blanket) ## Update the orientation
    #             # print('the new orientaion is:',new_orientation)
    #             return new_orientation
    #
    # def find_center_of_marker(self, marker_corners):
    #
    #     """find the center of each aruco markers"""
    #
    #     x_sum = marker_corners[0][0][0] + marker_corners[0][1][0] + marker_corners[0][2][0] + marker_corners[0][3][0]
    #     y_sum = marker_corners[0][0][1] + marker_corners[0][1][1] + marker_corners[0][2][1] + marker_corners[0][3][1]
    #     x_center = x_sum * .25
    #     y_center = y_sum * .25
    #     return (x_center, y_center)
    #
    # def check_members(self, marker_dict):
    #
    #     """Check which aruco was identifity"""
    #
    #     # if 43 in marker_dict:  ## and 47 in marker_dict
    #     #     return 1
    #     if 44 in marker_dict and 47 in marker_dict:  ## and 47 in marker_dict
    #         return 2
    #     elif 45 in marker_dict:  ## and 47 in marker_dict
    #         return 3
    #     elif 46 in marker_dict:  ## and 47 in marker_dict
    #         return 4
    #     else:
    #         return False
    #
    # def find_orientation(self, marker_dict):
    #
    #     """ find the card orientation """
    #
    #     key_list = self.markers.keys()
    #     if self.center is not None:
    #         if self.check_members(key_list) == 1:
    #              corner_2 = self.markers.get(43)
    #              self.orientation = self.find_dev(self.center, corner_2)
    #
    #         if self.check_members(key_list) == 2:
    #             corner_2 = self.markers.get(47)
    #             # corner_1= self.markers.get(47)
    #             self.orientation = self.find_dev(self.center, corner_2)
    #
    #         elif self.check_members(key_list) == 3:
    #             corner_2 = self.markers.get(45)
    #             self.orientation = self.find_dev(self.center, corner_2) #-np.pi/2
    #
    #         elif self.check_members(key_list) == 4:
    #             corner_2 = self.markers.get(46)
    #             self.orientation = self.find_dev(self.center, corner_2) #+ np.pi/2
    #     print ('the orinentation is:',self.orientation)
    #     return self.orientation
    #
    #
    def find_dev(self, q1, q2):
        """ Calculating the card angle in radians"""

        y = q2[1] - q1[1]
        x = q2[0] - q1[0]
        return math.atan2(y, x)
    def check_distance(self,epsilon):
        """ Check the Distance between the Center of card to destination point
        epslion is the resolution mistake allowed"""
        # self.center = list(self.center)
        distance = np.sqrt((self.center[0]-self.x_d)**2+(self.center[1]-self.y_d)**2)
        if distance < epsilon:
            self.error = distance
            # print("reached to destantion error is:{}".format(distance))
            return True
        return False


    def package_data(self):

        """ Package the data of the last iteration into dataframe row"""

        # print(self.orientation_list)
        # print(self.time_list)
        self.data.loc[self.data.shape[0], :] = np.array([np.array(self.path,dtype='uint64'),np.array(self.orientation_list),self.x_d,self.y_d,self.iteration,np.array(self.time_list),self.error], dtype ='object')

    def next_iteration(self):

        """Updating the iteration"""

        self.iteration = self.iteration+1
        # print("Itertaion Number is:{}".format(self.iteration)) ## Print the number of iteration to console

    def export_data(self):

        """ When finish export the data to pkl file"""

        with open('../../../data.pkl', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def angle_of_motor(self):
        dx = self.filter(self.last_dx,self.x_d - self.center[0])
        dy = self.filter(self.last_dy,self.y_d - self.center[1])

        new_angle = -(round(np.degrees(np.arctan2(dx,dy)))) +90
        if new_angle <0 :
            new_angle += 360
        return new_angle

    # Utility function to get the shortest way between two angles
    def shortest_way(self,num_1, num_2):
        if abs(num_1 - num_2) < 180:
            return num_2 - num_1
        else:
            if num_1 > num_2:
                return abs(num_1 - num_2 - 360)
            else:
                return abs(num_1 - num_2) - 360