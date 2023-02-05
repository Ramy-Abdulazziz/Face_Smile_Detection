import numpy as np
import cv2 as cv


class FaceDetectorHaar:

    def __init__(self,
                 path_face_casc,
                 scaling_factor=1.05,
                 nearest_neighbor=7,
                 frame = None,
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

        pts_lcorn_top = np.array([[x + (w//4), y],
                                  [x, y],
                                  [x, (y + (h//4))]],
                                 np.int32)

        pts_rcorn_top = np.array([[(x + w) - (w // 4), y],
                                  [(x + w), y],
                                  [x + w, (y + (h//4))]],
                                 np.int32)

        pts_lcorn_bot = np.array([[x, ((y + h) - (h//4))],
                                  [x, (y + h)],
                                  [(x + (w // 4)), (y + h)]],
                                 np.int32)

        pts_rcorn_bot = np.array([[x + w, ((y + h) - (h//4))],
                                  [x + w, y + h],
                                  [((x + w) - (w//4)), y + h]],
                                 np.int32)

        face_framer = [pts_lcorn_top, pts_lcorn_bot,
                       pts_rcorn_top, pts_rcorn_bot]

        face_framer = [x.reshape(-1, 1, 2) for x in face_framer]

        return (x, y, w, h, face_framer)

    def get_face_ROI(self, face_frame_coords:tuple, gray_frame):

        x, y = face_frame_coords[0], face_frame_coords[1]
        w, h = face_frame_coords[2], face_frame_coords[3]

        return gray_frame[y: y+h, x:x+w]


