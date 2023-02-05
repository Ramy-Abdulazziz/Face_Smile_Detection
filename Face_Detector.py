import Drawing_Functions as df
import cv2 as cv


class FaceDetectorHaar:

    def __init__(self,
                 path_face_casc,
                 scaling_factor=1.05,
                 nearest_neighbor=7,
                 frame=None,
                 ):

        self.path_face_casc = path_face_casc
        self.frame = frame

        self.scaling_factor = scaling_factor
        self.nearest_neighbor = nearest_neighbor

        self._face_cascade = cv.CascadeClassifier()

        self.init_cascade_classifiers()

    def setFrame(self, frame):
        self.frame = frame

    def init_cascade_classifiers(self):

        self._face_cascade.load(cv.samples.findFile(self.path_face_casc))

    def detect_faces(self):

        faces = self._face_cascade.detectMultiScale(self.frame,
                                                    self.scaling_factor,
                                                    self.nearest_neighbor)

        return faces

    def get_face_frame(self, face: tuple):

        x, y, w, h = face[0], face[1], face[2], face[3]

        face_framer = df.createSquareCornerFrame(x, y, w, h)
        
        return (x, y, w, h, face_framer)

    def get_face_ROI(self, face_frame_coords: tuple, gray_frame):

        x, y = face_frame_coords[0], face_frame_coords[1]
        w, h = face_frame_coords[2], face_frame_coords[3]

        return gray_frame[y: y+h, x:x+w]
