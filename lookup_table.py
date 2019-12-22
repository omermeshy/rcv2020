import cv2

import parameters


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, parameters.IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, parameters.IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # fix the frame with the parameters from the camera_calibration.py result
        frame = cv2.undistort(frame, parameters.CAMERA_MATRIX, parameters.DIST_COEFS, None)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
