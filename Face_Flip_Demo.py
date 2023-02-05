from Face_Detector import FaceDetectorHaar as fdh
import Drawing_Functions as df
import cv2 as cv


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

        faces = face_detector.detect_faces()

        for face in faces:

            (x, y, w, h, face_frame) = face_detector.get_face_frame(face)
            face_ROI = face_detector.get_face_ROI((x, y, w, h, face_frame), frame)

            frame[y:y+h, x:x+w] = cv.flip(face_ROI, -1)
            

        cv.imshow('Capturing -Face Flip', frame)

        if (cv.waitKey(1) == ord('q')) or (cv.waitKey(1) == ord('Q')):
            break

    closeWindows(capture)
