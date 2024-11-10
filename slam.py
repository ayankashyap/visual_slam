import cv2
import numpy as np
import matplotlib.pyplot as plt
from frontend import *
from visualize import *

if __name__ == "__main__":
    img1 = cv2.imread("03/left/000000.png")
    img2 = cv2.imread("03/left/000001.png")

    matcher = ORBmatcher()
    ret = matcher.match(img1, img2)
    pts1, pts2 = ret['pts']
    imshow("matches", drawmatches(img1, pts1, img2, pts2))

    F , (pts1, pts2) = compute_fundamental_matrix(pts1, pts2)
    pts1,pts2 = homo(pts1), homo(pts2)
    print(pts1.shape)
    verify_F_constraint(pts1, pts2, F)
    imshow("epipolarlines", draw_epipolar_lines(img1, pts1[:5], img2, pts2[:5], F))





    



