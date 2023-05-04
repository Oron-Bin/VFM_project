import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # define ArUco dictionary
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        # define ArUco parameters
        self.aruco_params = aruco.DetectorParameters_create()

        # create CvBridge instance
        self.bridge = CvBridge()

        # initialize variables for angle calculation
        self.prev_marker_pos = None
        self.curr_marker_pos = None

        # subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        # convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # detect ArUco markers in the image
        corners, ids, rejected = aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        # if at least two markers detected
        if len(corners) >= 2:
            # draw marker borders
            aruco.drawDetectedMarkers(cv_image, corners, ids)

            # get position of first two markers
            self.curr_marker_pos = np.squeeze(corners[:2])

            # if this is not the first iteration
            if self.prev_marker_pos is not None:
                # calculate vector connecting first two markers
                vec = self.curr_marker_pos[1] - self.curr_marker_pos[0]

                # calculate angle of vector
                angle = np.arctan2(vec[1], vec[0]) * 180 / np.pi

                # print ids of markers and angle
                self.get_logger().info(f"Calculating angle between markers {ids[0]} and {ids[1]}")
                self.get_logger().info(f"Angle of vector connecting first two markers: {angle} degrees")

            # update previous marker position
            self.prev_marker_pos = self.curr_marker_pos

        # display image with detected markers
        cv2.imshow('Aruco Detection', cv_image)
        cv2.waitKey(1)

    def on_shutdown(self):
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    aruco_detector = ArucoDetector()

    rclpy.spin(aruco_detector)

    aruco_detector.on_shutdown()

    aruco_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()