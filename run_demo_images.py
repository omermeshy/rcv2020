import glob
import imutils
import numpy as np
import cv2
import parameters
from find_distance_and_angle import find_robot_yawing_to_reflectors, find_v
from find_pair import find_pair
from video_from_the_robot import get_center_of_mass


def on_click(event, x, y, flags, param):
    """
    :param event: event type
    :param x: x pos
    :param y: y pos
    :param flags: not needed but must be param in the function
    :param param: not needed but must be param in the function
    :return: None
    """
    global lastY, lastX, first_click, min_hsv, max_hsv, image

    # check if the left button of the mouse clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # if this is the first press, save the x, y parameters to be able to use them to drw the rectangle
        if first_click:
            first_click = False
            lastY = y
            lastX = x

        # if this is the second press, get the max and min values inside the rectangle
        # draw rectangle with two circles on the pressed positions
        else:
            first_click = True

            # get the max and min values inside the rectangle
            roi = image[min(y, lastY):max(y, lastY), min(x, lastX):max(x, lastX)].copy()
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            (h, s, v) = cv2.split(roi)

            h_max_array.append(np.amax(h))
            s_max_array.append(np.amax(s))
            v_max_array.append(np.amax(v))

            h_min_array.append(np.amin(h))
            s_min_array.append(np.amin(s))
            v_min_array.append(np.amin(v))

            max_hsv = [np.amax(h_max_array), np.amax(s_max_array), np.amax(v_max_array)]
            min_hsv = [np.amin(h_min_array), np.amin(s_min_array), np.amin(v_min_array)]

            # draw the rectangle with the two circles on the pressed positions
            output = cv2.rectangle(image.copy(), (lastX, lastY), (x, y), (0, 255, 0), 2)
            output = cv2.circle(output, (lastX, lastY), 4, (0, 0, 255), -1)
            output = cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

            # print the minimum and maximum values
            # print("min hsv:  ", np.amin(h), np.amin(s), np.amin(v))
            # print("max hsv:  ", np.amax(h), np.amax(s), np.amax(v))

            # show the image
            cv2.imshow('frame', output)


def clear_noise(img):
    green = cv2.inRange(img, parameters.MIN_HSV, parameters.MAX_HSV)
    kernel = np.ones((7, 7), dtype=np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    # original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
    # print(green.shape, original_frame.shape)
    # numpy_hor_concat = np.concatenate((original_frame, green), axis=1)
    # cv2.imshow('original | in range', numpy_hor_concat)

    return green


def hsv_color_picker_demo():
    global lastY, lastX, first_click, min_hsv, max_hsv, image
    global h_max_array, s_max_array, v_max_array
    global h_min_array, s_min_array, v_min_array

    h_max_array = []
    s_max_array = []
    v_max_array = []

    h_min_array = []
    s_min_array = []
    v_min_array = []

    # the total minimum and maximum values (h, s, v)
    min_hsv = [-1, -1, -1]
    max_hsv = [-1, -1, -1]

    # אם זאת הלצחיצה הראשונה אז לא יצייר מלבן ויקח נתונים, אם לא, יצייר מלבן וייקח נתוני מינימום ומקסימום
    first_click = True

    # the x, y point for drawing the rectangle
    lastX = -1
    lastY = -1

    filenames = glob.glob(r"C:\drive\ImageProcessing\Images\reflectors\*")
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]

    # create a window and call on_click function if mouse made an action
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_click)

    for image in images:
        if image is not None:
            image = imutils.rotate(image, 15)
            cv2.imshow("frame", image)

            cv2.waitKey(0)

            image = imutils.rotate(image, -(15 * 2))
            cv2.imshow("frame", image)

            # get key pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if cv2.waitKey(0) == ord('q'):
                break

            # print the minimum and maximum values
            print("MIN_HSV = (" + str(tuple(min_hsv)) + ")")
            print("MAX_HSV = (" + str(tuple(max_hsv)) + ")")

    # cv2.destroyAllWindows()
    # cap.release()


def run_detection_code_on_test_images():
    filename = glob.glob(r"C:\drive\ImageProcessing\Images\reflectors\*")
    filename.sort()
    images = [cv2.imread(img) for img in filename]

    for image in images:
        if image is not None:
            image = cv2.resize(image, (parameters.IMAGE_WIDTH, parameters.IMAGE_HEIGHT))

            # fix the frame with the parameters from the camera_calibration.py result
            image = cv2.undistort(image, parameters.CAMERA_MATRIX, parameters.DIST_COEFS, None)

            original_frame = image

            # convert the color format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            green = clear_noise(image)

            image_with_reflectors_detection = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)

            # find the best pair in the image
            reflective1, reflective2 = find_pair(green)

            # if both of the shapes are shapes draw them on the image
            if (reflective1 is not None) and (reflective2 is not None):
                cv2.line(image_with_reflectors_detection, (160, 0), (160, 240), (255, 255, 255), 1)
                cv2.line(image_with_reflectors_detection, (0, 120), (320, 120), (255, 255, 255), 1)

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

            numpy_hor_concat = np.concatenate((original_frame, cv2.cvtColor(green, cv2.COLOR_GRAY2RGB),
                                               cv2.cvtColor(image_with_reflectors_detection, cv2.COLOR_BGR2RGB)),
                                              axis=1)
            cv2.imshow("""                           original                                                | 
                                             in range                                        |
                detected reflectors""", numpy_hor_concat)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(0)


if __name__ == '__main__':
    run_detection_code_on_test_images()
    # hsv_color_picker_demo()
