import cv2
import numpy as np


class Shape:
    #  Basic setting of general points
    top_right = []
    top_left = []
    bottom_right = []
    bottom_left = []

    # Initialize command, expecting a contour as an argument
    def __init__(self, contour):
        """
        :param contour: gets a contour when initialized
        Setting the top_right, top_left, bottom_right, bottom_left parameters
        The approxPollyDP works from top left to bottom right so
        the first point it finds is the highest and the second is the lowest.
        Afterwards we check if it's rotated left or right and assign
        The other points accordingly.
        """
        self.contour = contour
        epsilon = 0.02 * cv2.arcLength(self.contour, True)
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        if len(approx) == 4:
            if approx[1][0][1] > approx[3][0][1]:
                self.top_left = approx[0][0]
                self.top_right = approx[3][0]
                self.bottom_left = approx[1][0]
                self.bottom_right = approx[2][0]
            else:
                self.top_left = approx[1][0]
                self.top_right = approx[0][0]
                self.bottom_left = approx[2][0]
                self.bottom_right = approx[3][0]

    # Returns True if the function is above the avg points and False if not
    def calc_func(self):
        """
        :return: The slope of the function, the B of the function, the top average
        We get the average of the top side and bottom side of the rectangle
        and use the slope method to get it's slope. afterwards we calculate
        the b and return one point of the 2 averages.
        In this case, the top one.
        """
        avg_top = [(float(self.top_right[0]) + float(self.top_left[0])) / 2,
                   (float(self.top_right[1]) + float(self.top_left[1])) / 2]
        avg_bot = [(float(self.bottom_right[0]) + float(self.bottom_left[0])) / 2,
                   (float(self.bottom_right[1]) + float(self.bottom_left[1])) / 2]
        if avg_bot[1] == avg_top[1]:
            m = 0
        else:
            m = (avg_top[0] - avg_bot[0]) / (avg_top[1] - avg_bot[1])
        b = avg_top[1] - m * avg_top[0]
        return m, b, avg_top

    # A function that calculates the area of the contour
    def calc_area(self):
        """
        :return: The contour area of the approx shape
        """
        return cv2.contourArea(self.calc_contour())

    # A function the takes an array of shapes and determines which shape is the best to use
    def calc_relevance(self, shapes):
        """
        :param shapes: Gets a list of shapes to match the best one
        :return: The best shape
        The function goes through the list of shapes, for every shape it uses the
        Calc function on itself and on the other shape. Afterwards it checks
        If the functions collide over them or under them.
        If the collision is under them, they aren't compatible.
        The function also checks if the current best result is smaller
        than the current shape we are looking at, if it is, it's
        Replaced with the current shape as the best shape.
        """
        best = shapes[0]
        for i in shapes:
            m_self, b_self, top_self = self.calc_func()
            m_i, b_i, top_i = i.calc_func()
            if m_i != m_self and i.calc_area() > best.calc_area():
                meet = (b_i - b_self) / (m_self - m_i)
                if meet > top_i[1] and meet > top_self[1]:
                    best = i
                    # print("There is a best match!")
        # if best is None:
        #     return False
        return best

    # A function that returns the contour out of the corners
    def calc_contour(self):
        """
        :return: The approx contour
        """
        return np.array([self.top_right, self.top_left, self.bottom_left, self.bottom_right])

    def get_corners(self):
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left
