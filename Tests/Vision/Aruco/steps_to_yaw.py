import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def detect_circle_info(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (8, 8))
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.5, 1000, minRadius=50, maxRadius=500)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_center = (x, y)
            circle_radius = r

            return circle_center, circle_radius

    return None, None


def detect_aruco_centers(frame):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    aruco_centers = []
    if ids is not None:
        for i in range(len(ids)):
            aruco_center = np.mean(corners[i][0], axis=0)
            aruco_centers.append(aruco_center)
    # print(ids,aruco_centers)
    return aruco_centers ,ids


def calculate_angle(circle_center, aruco_center):
    dx = circle_center[0] - aruco_center[0]
    dy = circle_center[1] - aruco_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    return angle


def display_image(image, circle_center, circle_radius):
    if circle_center is not None and circle_radius is not None:
        cv2.circle(image, circle_center, circle_radius, (0, 255, 0), 4)
        cv2.circle(image, circle_center, 3, (0, 0, 255), -1)
    cv2.imshow('QueryImage', image)
    cv2.waitKey(1)

def ids_to_angle(ids,circle_center):
    # aruco_centers, ids = detect_aruco_centers(img)
    # circle_center, circle_radius = detect_circle_info(img)
    last_aruco_center = aruco_centers[-1]
    angle = calculate_angle(circle_center, last_aruco_center)
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
        print('The orientation is:', angle)
        return angle
    else:
        print('ids is none')
    #     angle = angle

def plot_angles(frame_numbers, angles):
    plt.plot(frame_numbers, angles)
    plt.xlabel('Frame')
    plt.ylabel('Angle')
    plt.ylim([0, 360])
    plt.title('Angle vs. Frame')
    plt.show()


# Main program

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

start_time = time.time()
times = []
angles = []
frame_numbers = []

while cam.isOpened():
    ret, img = cam.read()

    if ret:
        circle_center, circle_radius = detect_circle_info(img)
        aruco_centers, ids = detect_aruco_centers(img)
        # origin = tuple(circle_center)
        # scale = 50
        # # Define the endpoints of the X-axis and Y-axis relative to the origin
        # x_axis_end = (origin[0] + int(scale), origin[1])
        # y_axis_end = (origin[0], origin[1] + int(scale))
        #
        # # Draw coordinate system
        # cv2.line(img, origin, x_axis_end, (0, 0, 255), 2)  # X-axis (red)
        # cv2.line(img, origin, y_axis_end, (0, 255, 0), 2)

        if circle_center is not None and aruco_centers:
            origin = tuple(circle_center)
            scale = 50
            # Define the endpoints of the X-axis and Y-axis relative to the origin
            x_axis_end = (origin[0] + int(scale), origin[1])
            y_axis_end = (origin[0], origin[1] + int(scale))

            # Draw coordinate system
            cv2.line(img, origin, x_axis_end, (0, 0, 255), 2)  # X-axis (red)
            cv2.line(img, origin, y_axis_end, (0, 255, 0), 2)
            angle = ids_to_angle(ids, circle_center)
            angles.append(angle)
            current_time = time.time() - start_time
            times.append(current_time)
            frame_numbers.append(len(angles))

            display_image(img, circle_center, circle_radius)

        else:
            print("Warning")
            angle = 0
            print (angle)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Interrupted by user')
        break

    current_time = time.time() - start_time
    if current_time >= 20:
        break

cam.release()
cv2.destroyAllWindows()

plot_angles(frame_numbers, angles)