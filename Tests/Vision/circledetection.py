import numpy as np
import cv2

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
cam.set(cv2.CAP_PROP_AUTOFOCUS,0)



while cam.isOpened():
    # Capturing each frame of our video stream
    ret, Img = cam.read()
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY) #create an img with a gray scale
    gray_blurred = cv2.blur(gray, (8,8)) #blur the img
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ret:
        circles = cv2.HoughCircles(gray_blurred,
                   cv2.HOUGH_GRADIENT, 1.5, 1000,minRadius=20, maxRadius=200)
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                circle_center = (x, y)
                # print("x is",x,"y is", y,"radius is", r)
                # print(circles[0][1])
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(Img, (x, y), r, (0, 255 ,0), 4) # the color is in RGB and the last parameter is the thickness
                cv2.rectangle(Img, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1) #thickness of -1 means to fill the rectangle
            if len(ids) > 0:
                # Calculate the center of the ArUco marker
                aruco_center = np.mean(corners[0][0], axis=0)

                # Draw a small dot at the center of the ArUco marker
                cv2.circle(Img, tuple(aruco_center.astype(int)), 3, (255, 0, 0), -1)

                # Calculate the angle between the centers of the circle and the ArUco marker
                dx = circle_center[0] - aruco_center[0]
                dy = circle_center[1] - aruco_center[1]
                angle = np.degrees(np.arctan2(dy, dx))

                print("Angle:", angle)

        cv2.imshow('QueryImage', Img)
        cv2.waitKey(1) #shows continuous live video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interupt by user')
            break