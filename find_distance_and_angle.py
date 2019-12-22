import cv2
from shape import Shape
import parameters


def find_angle_from_px(obj_size_in_px, img_width):
    """
    calculate angle from px
    :param obj_size_in_px: the size in pixels of the detected object
    :param img_width: the image width in pixels
    :return: returns the angle of the object from the center of the image.
             the center is 0 degrees and every this left to it is negative: [ - 0 + ]
    """

    # calc the pixel to degrees
    return (obj_size_in_px - img_width) / parameters.PIXEL_TO_DEGREES_RATIO


def find_v(rectangle_left: Shape, rectangle_right: Shape):
    """
    find the vertical line from the center of the given objects.
    :param rectangle_left: shape object
    :param rectangle_right: shape object
    :return: returns the x and y of the two points of line
    """
    left_top_left, left_top_right, left_bottom_right, left_bottom_left = rectangle_left.get_corners()
    right_top_left, right_top_right, right_bottom_right, right_bottom_left = rectangle_right.get_corners()

    top_avg_x = (left_top_left[1] + left_top_right[1] + right_top_left[1] + right_top_right[1]) / 4
    top_avg_y = (left_top_left[0] + left_top_right[0] + right_top_left[0] + right_top_right[0]) / 4

    bottom_avg_x = (left_bottom_left[1] + left_bottom_right[1] + right_bottom_left[1] + right_bottom_right[1]) / 4
    bottom_avg_y = (left_bottom_left[0] + left_bottom_right[0] + right_bottom_left[0] + right_bottom_right[0]) / 4

    return int(top_avg_x), int(top_avg_y), int(bottom_avg_x), int(bottom_avg_y)


def find_robot_yawing_to_reflectors(shape1, shape2):
    """
    find the yaw of the robot.
    calculate the average distance between the center of shape1 and shape2 and than divide it by the
    shape1: shape object
    shape2: shape object
    :return: the yaw of the robot
    """

    contours_shape1 = shape1.calc_contour()
    contours_shape2 = shape2.calc_contour()

    # compute the center of the contour
    m = cv2.moments(contours_shape1)
    if m["m00"] != 0:
        c_x1 = int(m["m10"] / m["m00"])
        # c_y1 = int(m["m01"] / m["m00"])
    else:
        c_x1, c_y1 = 0, 0

    # compute the center of the contour
    m = cv2.moments(contours_shape2)
    if m["m00"] != 0:
        c_x2 = int(m["m10"] / m["m00"])
        # c_y2 = int(m["m01"] / m["m00"])
    else:
        c_x2, c_y2 = 0, 0

    yaw = (c_x1 + c_x2) / 2
    yaw = (yaw / parameters.PIXEL_TO_DEGREES_RATIO) - 26
    return yaw
