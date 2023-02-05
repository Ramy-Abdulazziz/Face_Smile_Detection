import numpy as np
import cv2 as cv
from Face_Smile_Detector import FaceSmileDetectorHaar as fsdh


def detectAndWait(capture: cv.VideoCapture(), face_smile_detector: fsdh):

    ret, frame = capture.read()
    face_smile_detector.setFrame(frame)

    if not ret:
        print("Error reading frame - check video source")
        pass

    face_smile_detector.detectionSequence()
    frame = face_smile_detector.getFrame()

    cv.imshow('Capturing -Face and Smile Detection', frame)


def initInputSource(source_num):
    return cv.VideoCapture(source_num)


def closeWindows(capture: cv.VideoCapture):
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    path_to_face_cascade = "C:\\Users\\ramya\\OPEN-CV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    path_to_smile_casade = "C:\\Users\\ramya\\OPEN-CV\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml"

    capture = initInputSource(0)
    face_smile_detector = fsdh(path_to_face_cascade, path_to_smile_casade,
                               (252, 3, 44), (3, 28, 252), 3, 2, 1.05, 1.3, 5, 22)

    while True:

        detectAndWait(capture, face_smile_detector)

        if cv.waitKey(1) == ord('q'):
            break

    closeWindows(capture)
