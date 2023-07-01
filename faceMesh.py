import cv2
import mediapipe as mp
import time

class FaceMesh:
    
    def __init__(self, mode=False, maxFaces=3, refineLandmarks=False, detectConf=0.5, trackConf=0.5):
        
        self.mode=mode
        self.maxFaces=maxFaces
        self.refineLandmarks=refineLandmarks
        self.detectConf=detectConf
        self.trackConf=trackConf

        self.mpFacemesh = mp.solutions.face_mesh
        self.facemesh = self.mpFacemesh.FaceMesh(self.mode, self.maxFaces, self.refineLandmarks, self.detectConf, self.trackConf)
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawingSpec = self.drawing_utils.DrawingSpec(thickness = 1, circle_radius=1)


    def createFaceMesh(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facemesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for landmark in self.results.multi_face_landmarks:
                if draw:
                    self.drawing_utils.draw_landmarks(img, landmark, self.mpFacemesh.FACEMESH_CONTOURS, self.drawingSpec, self.drawingSpec)
        
        return img

    
    def getPosition(self, img, faceNo=0, draw=False):

        lmList = []
        if self.results.multi_face_landmarks:
            faceLandmark = self.results.multi_face_landmarks[faceNo]
            for idx, lm in enumerate(faceLandmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList, img
