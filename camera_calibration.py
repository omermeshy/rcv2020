# import necessary libraries
import cv2
import numpy as np
import glob

PATTERN_SIZE = (9, 6)
SQUARE_SIZE = float(25)

pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2) * SQUARE_SIZE

obj_points = []
img_points = []
h, w, = 0, 0

images = glob.glob('C:/Users/Alut/PycharmProjects/season2020_practice/outputVideo/*.jpg')

for fname in images:
    image = cv2.imread(fname)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = imageGray.shape

    found, corners = cv2.findChessboardCorners(imageGray, PATTERN_SIZE, None)

    if found:
        cv2.drawChessboardCorners(image, PATTERN_SIZE, corners, found)
        cv2.imshow("Chess", image)
        cv2.waitKey(100)
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

# get the parameters of the calibration
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

# print the parameters
print("RMS:", rms)
print("---------------")
print("Camera matrix:\n", camera_matrix)
print("---------------")
print("Distortion:\n", dist_coefs)
print("---------------")

# show the before and after images
for frame in images:
    image = cv2.imread(frame)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = imageGray.shape

    dst = cv2.undistort(image, camera_matrix, dist_coefs, None)
    numpy_hor_concat = np.concatenate((image, dst), axis=1)
    cv2.imshow('Numpy Vertical Concat', numpy_hor_concat)
    cv2.waitKey(0)
