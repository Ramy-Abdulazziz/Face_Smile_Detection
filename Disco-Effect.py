from Face_Detector import FaceDetectorHaar as fdh
import Drawing_Functions as df
import cv2 as cv
import numpy as np
import itertools as it


def initInputSource(source_num):
    return cv.VideoCapture(source_num)


def closeWindows(capture: cv.VideoCapture):
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    path_to_face_cascade = f"{cv.data.haarcascades}haarcascade_frontalface_default.xml"

    capture = initInputSource(0)
    face_detector = fdh(path_to_face_cascade)

   

    while True:

        ret, frame = capture.read()
        face_detector.setFrame(frame)

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.equalizeHist(gray_frame)
        

        if not ret:
            print("Error reading frame - check video source")
            pass

        (ih, iw, ic) = frame.shape
        faces = face_detector.detect_faces()
        new_frame = np.zeros((ih, iw, ic), np.uint8)

        for face in faces:

            (x, y, w, h, face_frame) = face_detector.get_face_frame(face)
            face_ROI = face_detector.get_face_ROI((x, y, w, h, face_frame), frame)

            y2, x2, c  =0, 0, 0 

            while (y2 + h) < ih:

                new_frame[y2:y2+h, x2:x2+w, c]  = face_ROI[:,: , c]
                x2 = x2 + w
                
                c = c + 1

                if c == 3:
                    c = 0
                
                if(x2 + w >= iw):
                    x2 = 0
                    y2 = y2 + h
                
            
        # cv.imshow('Capturing -old frfame', frame)
        cv.imshow('Capturing -new frame', new_frame)


        if (cv.waitKey(1) == ord('q')) or (cv.waitKey(1) == ord('Q')):
            break

    closeWindows(capture)
