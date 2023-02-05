import Drawing_Functions as df
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

        return df.createSquareCornerFrame(x,y,w,h)
