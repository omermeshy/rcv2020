import cv2
import numpy as np

# the minimum and maximum values (h, s, v)
min_hsv = [256, 256, 256]
max_hsv = [-1, -1, -1]

# אם זאת הלצחיצה הראשונה אז לא יצייר מלבן ויקח נתונים, אם לא, יצייר מלבן וייקח נתוני מינימום ומקסימום
first_click = True

# the x, y point for drawing the rectangle
lastX = -1
lastY = -1


def on_click(event, x, y, flags, param):
    """
    :param event: event type
    :param x: x pos
    :param y: y pos
    :param flags: not needed but must be param in the function
    :param param: not needed but must be param in the function
    :return: None
    """
    global lastY, lastX, first_click, min_hsv, max_hsv

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
            roi = frame[min(y, lastY):max(y, lastY), min(x, lastX):max(x, lastX)].copy()
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            (h, s, v) = cv2.split(roi)
            min_hsv = [min(np.amin(h), min_hsv[0]), min(np.amin(s), min_hsv[1]), min(np.amin(v), min_hsv[2])]
            max_hsv = [max(np.amax(h), max_hsv[0]), max(np.amax(s), max_hsv[1]), max(np.amax(v), max_hsv[2])]

            # draw the rectangle with the two circles on the pressed positions
            output = cv2.rectangle(frame.copy(), (lastX, lastY), (x, y), (0, 255, 0), 2)
            output = cv2.circle(output, (lastX, lastY), 4, (0, 0, 255), -1)
            output = cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

            # show the image
            cv2.imshow('frame', output)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # create a window and call on_click function if mouse made an action
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_click)

    while True:
        _, frame = cap.read()
        # frame = cv2.flip(frame, 180)
        cv2.imshow("frame", frame)

        # get key pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord(' '):
            if cv2.waitKey(0) == ord('q'):
                break

    # print the minimum and maximum values
    print("MIN_HSV = " + str(tuple(min_hsv)))
    print("MAX_HSV = " + str(tuple(max_hsv)))

    # cv2.destroyAllWindows()
    # cap.release()
