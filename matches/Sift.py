# goal:获得SIFT初始特征匹配
import cv2
import numpy as np


def get_initial_matches(img1, img2, rt):
    matcher, kp1, kp2, desc1, desc2 = siftTest(img1, img2)
    src_pts_RT_81, dst_pts_RT_81, good_num_RT_81, scores_RT81 = get_pts(matcher, kp1, kp2, desc1, desc2, rt)
    return src_pts_RT_81, dst_pts_RT_81, good_num_RT_81, scores_RT81

def siftTest(img1, img2):
    # 	contrastThreshold： 对比度阈值，用于过滤低对比度区域中的特征点。阈值越大，检测器产生的特征越少。 (sift论文用的0.03，nOctaveLayers若为3， 设置参数为0.09，实际值为：contrastThreshold/nOctaveLayers)
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.05)
    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=0.04)
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    return bf, kp1, kp2, desc1, desc2


def get_pts(matcher, kp1, kp2, desc1, desc2, b):
    raw_matches = matcher.knnMatch(desc1, desc2, 2)  # 2
    good = []
    scores = []
    for m, n in raw_matches:
        if m.distance < b * n.distance:
            scores.append(m.distance / n.distance)
            # if m.distance < 300:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    npts1 = []
    npts2 = []
    for keypoint in kp1:
        npts1.append((keypoint.pt[0], keypoint.pt[1]))
    for keypoint in kp2:
        npts2.append((keypoint.pt[0], keypoint.pt[1]))
    src_pts_left = np.float32([npts1[m.queryIdx] for m in good])  # (x,y)格式的内点
    dst_pts_right = np.float32([npts2[m.trainIdx] for m in good])
    return src_pts_left, dst_pts_right, len(good), scores



