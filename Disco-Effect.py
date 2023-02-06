from Face_Detector import FaceDetectorHaar as fdh
import cv2 as cv
import numpy as np

def initInputSource(source_num):
    return cv.VideoCapture(source_num)


def closeWindows(capture: cv.VideoCapture):
    capture.release()
    cv.destroyAllWindows()


def calc_frame_square(frame: np.array, sqr_dim: tuple):

    (ih, iw, ic) = frame.shape

    nh = ih//sqr_dim[0]
    nw = iw//sqr_dim[1]

    dimensions = (nw, nh)

    return dimensions


if __name__ == "__main__":

    path_to_face_cascade = f"{cv.data.haarcascades}haarcascade_frontalface_default.xml"

    capture = initInputSource(0)
    face_detector = fdh(path_to_face_cascade)

    frame_list = list()

    while True:

        ret, frame = capture.read()
        face_detector.setFrame(frame)

        (ih, iw, ic) = frame.shape
        new_frame = np.zeros((ih, iw, ic), np.uint8)
        dimensions = calc_frame_square(new_frame, (4, 4))

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.equalizeHist(gray_frame)

        if not ret:
            print("Error reading frame - check video source")
            pass

        faces = face_detector.detect_faces()

        for face in faces:

            (x, y, w, h, face_frame) = face_detector.get_face_frame(face)

            face_ROI = face_detector.get_face_ROI(
                (x, y, w, h, face_frame), frame)

            resized = cv.resize(face_ROI, dimensions,
                                interpolation=cv.INTER_AREA)

            h = dimensions[1]
            w = dimensions[0]

            y2, x2, c = 0, 0, 0

            while (y2 + h) <= ih:

                new_frame[y2:y2+h, x2:x2+w, c] = resized[:, :, c]
                x2 = x2 + w

                c = c + 1

                if c == 3:
                    c = 0

                if (x2 + w > iw):
                    x2 = 0
                    y2 = y2 + h

        if len(faces) == 0:
            cv.imshow('Capturing -new frame', frame)
        else:
            cv.imshow('Capturing -new frame', new_frame)

        if (cv.waitKey(1) == ord('q')) or (cv.waitKey(1) == ord('Q')):
            break

    closeWindows(capture)
