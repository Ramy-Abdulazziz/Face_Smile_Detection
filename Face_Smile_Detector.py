import cv2 as cv
from Face_Detector import FaceDetectorHaar as fdh
from Smile_Detector import SmileDetectorHaar as sdh


class FaceSmileDetectorHaar:

    def __init__(self,
                 path_to_face_cascade,
                 path_to_smile_casade,
                 face_detect_color=(255, 0, 255),
                 smile_detect_color=(255, 255, 0),
                 face_detect_thickness: int = 5,
                 smile_detect_thickness: int = 3,
                 face_detect_scale_factor=1.05,
                 smile_detect_scale_factor=1.3,
                 face_detect_nearest_neighbor=6,
                 smile_detect_nearest_neighbor=25,

                 ):

        self.face_detector = fdh(
            path_to_face_cascade, face_detect_scale_factor, face_detect_nearest_neighbor)
        self.smile_detector = sdh(
            path_to_smile_casade, smile_detect_scale_factor, smile_detect_nearest_neighbor)

        self.face_detect_color = face_detect_color
        self.smile_detect_color = smile_detect_color

        self.face_detect_thickness = face_detect_thickness
        self.smile_detect_thickness = smile_detect_thickness

        self.frame = None
        self.gray_frame = None

    def setFrame(self, frame):

        self.frame = frame
        self.face_detector.setFrame(frame)
        self.smile_detector.setFrame(frame)

    def getFrame(self):
        return self.frame

    def grayEqualizeHist(self):

        gray_frame_eq = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.gray_frame = cv.equalizeHist(gray_frame_eq)

    def detectFaceAndDraw(self, face_coords_frame: tuple):

        face_frame = face_coords_frame[4]
        self.frame = cv.polylines(
            self.frame, face_frame, False, self.face_detect_color, self.face_detect_thickness)

    def detectSmilesAndDraw(self, frame, smile, face_coords_frame: tuple):

        face_x = face_coords_frame[0]
        face_y = face_coords_frame[1]

        smile_frame = self.smile_detector.get_smile_frame(
            smile, face_x, face_y)
        self.frame = cv.polylines(
            frame, smile_frame, False, self.smile_detect_color, self.smile_detect_thickness)

    def detectionSequence(self):

        self.grayEqualizeHist()
        faces = self.face_detector.detect_faces()

        for face in faces:

            face_coords_frame = self.face_detector.get_face_frame(face)
            self.detectFaceAndDraw(face_coords_frame)

            face_ROI = self.face_detector.get_face_ROI(
                face_coords_frame, self.gray_frame)
            self.smile_detector.setFaceROI(face_ROI)
            smiles = self.smile_detector.detect_smiles()

            for smile in smiles:

                self.detectSmilesAndDraw(self.frame, smile, face_coords_frame)
