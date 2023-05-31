import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

start_time = time.time()
times = []
angles = []

while cam.isOpened():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    ret, Img = cam.read()
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (8, 8))
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ret:
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.5, 1000, minRadius=50, maxRadius=300)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                circle_center = (x, y)
                angle = 0
                if ids is not None:
                    for i in range(len(ids)):
                        aruco_center = np.mean(corners[i][0], axis=0)
                    dx = circle_center[0] - aruco_center[0]
                    dy = circle_center[1] - aruco_center[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    axis_x = (circle_center[0] - 50, circle_center[1])
                    axis_y = (circle_center[0], circle_center[1] - 50)
                    cv2.arrowedLine(
                        Img,
                        tuple(circle_center),
                        axis_x,
                        (0, 0, 0),
                        2,
                        tipLength=0.2
                    )
                    cv2.arrowedLine(
                        Img,
                        tuple(circle_center),
                        axis_y,
                        (0, 0, 0),
                        2,
                        tipLength=0.2
                    )

                    if ids[-1] == 43:
                        angle = np.degrees(np.arctan2(dy, dx))
                        cv2.arrowedLine(
                            Img,
                            tuple(circle_center),
                            tuple(aruco_center),
                            (0, 0, 255),
                            2,
                            tipLength=0.2
                        )
                    elif ids[-1] == 44:
                        angle += 180
                    elif ids[-1] == 45:
                        angle += 90
                    elif ids[-1] == 46:
                        angle -= 90
                    if angle < 0:
                        angle += 360
                    print('The orientation is:', angle)
                    current_time = time.time() - start_time
                    times.append(current_time)
                    angles.append(angle)
                else:
                    print('ids is none')
                    angle = angle

                cv2.circle(Img, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(Img, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)
        cv2.imshow('QueryImage', Img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interrupted by user')
            break

        current_time = time.time() - start_time
        if current_time >= 10:
            break

cam.release()
cv2.destroyAllWindows()

plt.plot(times, angles)
plt.xlabel('Time (s)')
plt.ylabel('Angle')
plt.ylim([0,360])
plt.title('Angle vs. Time')
plt.show()
