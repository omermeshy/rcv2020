import numpy as np
import cv2
import parameters
import matplotlib.pyplot as plt
import time
from find_distance_and_angle import *
from find_pair import find_pair
from align_light_reflectors import align_shape


def draw_graph(x, y, avg):
    """
    draw graph of the points to see the progress
    :param avg: average time per frame
    :param x: list of fill x (frame number)
    :param y: list of fill y (millis)
    :return: None
    """

    # plotting the line 1 points
    plt.plot(x, y, label="time", marker='o', markersize=2, markerfacecolor='red')
    # plotting the line 2 points
    plt.plot([0, len(x)], [0, 0], label="time = 0")
    # plotting the line 3 points
    plt.plot([0, len(x)], [avg, avg], label="average")

    plt.grid(True)

    # naming the x axis
    plt.xlabel('frame')
    # naming the y axis
    plt.ylabel('millis')

    # giving a title to my graph
    plt.title('time per frame')

    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.show()


def clear_noise(image):
    """
    clear the noise on the image
    :param image: hsv image
    :return: cleared image
    """
    green = cv2.inRange(image, parameters.MIN_HSV, parameters.MAX_HSV)
    kernel = np.ones((7, 7), dtype=np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    return green


def get_center_of_mass(reflective_contour):
    """
    get the center of a shape
    :param reflective_contour: contour of of a shape
    :return: point of the center
    """
    # compute the center of the contour
    m = cv2.moments(reflective_contour)

    # calc the center of the shape
    if m["m00"] != 0:
        c_x = int(m["m10"] / m["m00"])
        c_y = int(m["m01"] / m["m00"])
    else:
        c_x, c_y = 0, 0

    return c_x, c_y


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, parameters.IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, parameters.IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    frame_number = []
    time_list = []
    frame_num = 0

    while cap.isOpened():
        # get the first time of the current taken frame
        first_time = int(round(time.time() * 1000))

        # Capture frame-by-frame
        ret, frame = cap.read()

        # fix the frame with the parameters from the camera_calibration.py result
        frame = cv2.undistort(frame, parameters.CAMERA_MATRIX, parameters.DIST_COEFS, None)

        # and flip the image by 180 degrees
        # frame = cv2.flip(frame, 180)
        original_frame = frame

        # convert the color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green = clear_noise(frame)
        image_with_reflectors_detection = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)

        # find the best pair in the image
        reflective1, reflective2 = find_pair(green)

        cv2.line(image_with_reflectors_detection, (160, 0), (160, 240), (255, 255, 255), 1)
        cv2.line(image_with_reflectors_detection, (0, 120), (320, 120), (255, 255, 255), 1)

        # if both of the shapes are shapes draw them on the image
        if (reflective1 is not None) and (reflective2 is not None):
            reflective1_contour = reflective1.calc_contour()
            reflective2_contour = reflective2.calc_contour()

            image_with_reflectors_detection = cv2.drawContours(image_with_reflectors_detection, [reflective1_contour],
                                                               -1, (255, 255, 255), -1)
            image_with_reflectors_detection = cv2.drawContours(image_with_reflectors_detection, [reflective1_contour],
                                                               -1, (0, 0, 255), 2)

            image_with_reflectors_detection = cv2.drawContours(image_with_reflectors_detection, [reflective2_contour],
                                                               -1, (255, 255, 255), -1)
            image_with_reflectors_detection = cv2.drawContours(image_with_reflectors_detection, [reflective2_contour],
                                                               -1, (255, 0, 0), 2)
            # draw a circle in the center of mass of the first shape
            c_x1, c_y1 = get_center_of_mass(reflective1_contour)
            cv2.circle(image_with_reflectors_detection, (c_x1, c_y1), 3, (0, 255, 0), -1)

            # draw a circle in the center of mass of the second shape
            c_x2, c_y2 = get_center_of_mass(reflective2_contour)
            cv2.circle(image_with_reflectors_detection, (c_x2, c_y2), 3, (0, 255, 0), -1)

            # draw lines between the centers of the shapes
            top_avg_x, top_avg_y, bottom_avg_x, bottom_avg_y = find_v(reflective1, reflective2)
            cv2.line(image_with_reflectors_detection, (top_avg_y, top_avg_x),
                     (bottom_avg_y, bottom_avg_x), (0, 0, 255), 3)

            cv2.line(image_with_reflectors_detection, (c_x1, c_y1),
                     (c_x2, c_y2), (0, 255, 255), 3)

            # draw point on the middle of the shapes (cross point of the lines from the centers)
            cv2.circle(image_with_reflectors_detection,
                       (int((top_avg_y + bottom_avg_y) / 2), int((top_avg_x + bottom_avg_x) / 2)), 5, (255, 255, 255))

            # calc the pixel to degrees
            x = int((top_avg_y + bottom_avg_y) / 2)
            print("x =", x, "angle =", (x - 160) / parameters.PIXEL_TO_DEGREES_RATIO)

        # create one big image
        numpy_hor_concat = np.concatenate((original_frame, cv2.cvtColor(green, cv2.COLOR_GRAY2RGB),
                                           cv2.cvtColor(image_with_reflectors_detection, cv2.COLOR_BGR2RGB)),
                                          axis=1)
        cv2.imshow("""                           original                                                | 
                                         in range                                        |
            detected reflectors""", numpy_hor_concat)

        frame_num += 1
        frame_number.append(frame_num)
        time_list.append(int(round(time.time() * 1000)) - first_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    total_time = 0

    # pop the first item because it always takes much more time
    time_list.pop(0)
    frame_number.pop(0)

    # calc the total time
    for specific_time in time_list:
        total_time += specific_time
    print("average time per frame: ", total_time / frame_num, "millis")

    # draw a times graph
    draw_graph(frame_number, time_list, total_time / frame_num)


if __name__ == '__main__':
    main()
