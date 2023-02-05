import numpy as np
import cv2 as cv


class SmileDetectorHaar:

    def __init__(self,
                 path_smile_casc,
                 scaling_factor=1.8,
                 nearest_neighbor=20,
                 frame=None
                 ):

        self.path_smile_casc = path_smile_casc
        self.frame = frame

        self.scaling_factor = scaling_factor
        self.nearest_neighbor = nearest_neighbor

        self._smile_cascade = cv.CascadeClassifier()
        self.init_cascade_classifiers()

        self.faceROI = None

    def setFrame(self, frame):
        self.frame = frame

    def setFaceROI(self, faceROI):
        self.faceROI = faceROI

    def init_cascade_classifiers(self):

        self._smile_cascade.load(cv.samples.findFile(self.path_smile_casc))

    def detect_smiles(self):

        smiles = self._smile_cascade.detectMultiScale(self.faceROI,
                                                      self.scaling_factor,
                                                      self.nearest_neighbor)
        return smiles

    def get_smile_frame(self, smiles: tuple, face_x, face_y):

        x, y = smiles[0] + face_x, smiles[1] + face_y
        w, h = smiles[2], smiles[3]

        spts_lcorn_top = np.array([[x + (w//4), y],
                                   [x, y],
                                   [x, y + (h//4)]],
                                  np.int32)

        spts_lcorn_bot = np.array([[x, (y + h) - (h//4)],
                                   [x, (y + h)],
                                   [(x + (w//4)), (y + h)]],
                                  np.int32)

        spts_rcorn_top = np.array([[x + w - (w//4), y],
                                   [x + w, y],
                                   [x + w, y + (h//4)]],
                                  np.int32)

        spts_rcorn_bot = np.array([[x + w - (w//4), y + h],
                                   [x + w, y + h],
                                   [x + w, y + h - (h//4)]],
                                  np.int32)

        smile_frame = [spts_lcorn_top, spts_lcorn_bot,
                       spts_rcorn_top, spts_rcorn_bot]
        smile_frame = [x.reshape((-1, 1, 2)) for x in smile_frame]

        return smile_frame
