import cv2
import numpy as np

# Load the ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

# Create the ArUco parameters
aruco_params = cv2.aruco.DetectorParameters_create()

# Load the image
image = cv2.imread("image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the ArUco markers
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

# Detect the circle object
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=5, maxRadius=100)

if circles is not None:
    # Convert the circle parameters to integers
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        # Draw the circle on the image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # Draw a small dot at the center of the circle
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # Calculate the center of the circle
        circle_center = (x, y)

        # Find the center of the ArUco marker
        for i in range(len(ids)):
            # Calculate the center of the ArUco marker
            aruco_center = np.mean(corners[i][0], axis=0)

            # Draw a small dot at the center of the ArUco marker
            cv2.circle(image, tuple(aruco_center.astype(int)), 3, (255, 0, 0), -1)

            # Calculate the angle between the centers of the circle and the ArUco marker
            dx = circle_center[0] - aruco_center[0]
            dy = circle_center[1] - aruco_center[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # Print the angle
            print("Angle:", angle)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()