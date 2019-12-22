import cv2
from shape import Shape
import numpy as np


def find_pair(image):
    count = 0
    count_biggest = 0
    img1, contours, heir = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("found contours:", len(contours))
    if len(contours) > 1:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for j in range(len(contours)):
            shapes = []
            biggest = contours[0]
            for i in contours:
                epsilon = 0.02 * cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, epsilon, True)
                output = cv2.drawContours(output, [i], -1, (255, 255, 255), -1)
                output = cv2.drawContours(output, [i], -1, (0, 0, 255), 2)
                if len(approx) > 4:
                    min_y = approx[0][0]
                    min_x = approx[0][0]
                    max_y = approx[0][0]
                    max_x = approx[0][0]
                    for m in range(len(approx)):
                        if approx[m][0][0] > max_x[0]:
                            max_x = approx[m][0]
                        if approx[m][0][0] < min_x[0]:
                            min_x = approx[m][0]
                        if approx[m][0][1] > max_y[1]:
                            max_y = approx[m][0]
                        if approx[m][0][1] < min_y[1]:
                            min_y = approx[m][0]
                    approx = cv2.approxPolyDP(np.array([min_y, min_x, max_y, max_x]), epsilon, True)
                if len(approx) == 4:
                    shapes.append(Shape(approx))
                    if cv2.contourArea(approx) > cv2.contourArea(biggest):
                        biggest = i
                        count_biggest = count
                    count += 1
            cv2.imshow("image with contours", output)
            if len(shapes) > 1:
                return shapes[count_biggest], shapes[count_biggest].calc_relevance(shapes)
            else:
                return None, None
    else:
        return None, None
