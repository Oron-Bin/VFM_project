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

    if ret: #a boolian that tells if there is a call from the camera
        circles = cv2.HoughCircles(gray_blurred,
                   cv2.HOUGH_GRADIENT, 1.5, 1000,minRadius=50, maxRadius=300)
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                print("x is",x,"y is", y,"radius is", r)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(Img, (x, y), r, (0, 255 ,0), 4) # the color is in RGB and the last parameter is the thickness
                cv2.rectangle(Img, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1) #thickness of -1 means to fill the rectangle
        cv2.imshow('QueryImage', Img)
        cv2.waitKey(1) #shows continuous live video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Interupt by user')
            break