import cv2
import numpy as np
import math

def homo(pts): return np.hstack((pts, np.ones((len(pts), 1))))

def dehomo(pts): return pts[:, :1] / pts[:, 2][:, np.newaxis]


class ORBmatcher:
    def __init__(self, ratio_threshold=0.7):
        self.featurizer = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_threshold = ratio_threshold
        
    def extract(self, img):
        # detect good points
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        # extract features
        kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=20) for p in pts]
        kps, des = self.featurizer.compute(img, kps)
        return  kps, des

    def match(self, img1, img2):
        kp1, des1 = self.extract(img1)
        kp2, des2 = self.extract(img2)
        
        matches = []
        for m, n in self.matcher.knnMatch(des1, des2, k=2):
            if m.distance < (self.ratio_threshold * n.distance):
                matches.append(m)
        
        assert len(matches) >= 8, "Not enough points for essential matrix"

        pts1 = np.int32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.int32([kp2[m.trainIdx].pt for m in matches])

        return {'pts': (pts1, pts2), 'kps': (kp1, kp2), 'des': (des1, des2)}


def compute_fundamental_matrix(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_LMEDS)
    return F, (pts1[mask.ravel() == 1], pts2[mask.ravel() == 1])









    