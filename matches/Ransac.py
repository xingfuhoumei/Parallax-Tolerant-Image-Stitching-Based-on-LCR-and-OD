# goal:利用RANSAC进行匹配
import cv2
import numpy as np
import all


def get_RANSAC_points(img1, img2, src_pts_RT_8, dst_pts_RT_8, good_num_RT_8):
    # 整体思路，利用右图原点序号判断 变换后左图序号重新分块（这个时候变换前后的块是一致的）
    # fixme: 为什么设置残差会比真实残差大?
    M, mask = cv2.findHomography(src_pts_RT_8, dst_pts_RT_8, method=cv2.RANSAC, ransacReprojThreshold=4)
    # 左图原点,左图变换后点,右图原点,RANSAC匹配图
    src_pts_left, src_pts_left_moved, dst_pts_right_moved, matches1, all.cnt = RANSAC_show(good_num_RT_8, src_pts_RT_8,
                                                                                           dst_pts_RT_8, mask, M, img1,
                                                                                           img2)
    return src_pts_left, src_pts_left_moved, dst_pts_right_moved, matches1, M


def RANSAC_show(good, src_pts_left, dst_pts_right, mask, M, img1, img2):
    matchesMask = mask.ravel().tolist()
    cnt = 0
    src_pts_left_ = []
    dst_pts_right_ = []
    for i in range(good):  # 把经过ransac的点筛出来
        if matchesMask[i] == 1:
            cnt += 1
            src_pts_left_.append(src_pts_left[i])
            dst_pts_right_.append(dst_pts_right[i])
    src_pts_left_ = np.array(src_pts_left_)
    dst_pts_right_ = np.array(dst_pts_right_)

    src_pts_left_moved = np.zeros((cnt, 2))
    src_pts_left_moved2 = np.zeros((3, cnt))
    dst_pts_right_moved = dst_pts_right_
    src_pts_left_2 = np.array([src_pts_left_[:, 0],
                               src_pts_left_[:, 1],
                               np.ones(len(src_pts_left_))])
    for i in range(cnt):
        src_pts_left_moved2[:, i] = M @ src_pts_left_2[:, i]
        src_pts_left_moved2[:, i] = src_pts_left_moved2[:, i] / src_pts_left_moved2[2, i]
    src_pts_left_moved2 = np.transpose(src_pts_left_moved2[0:2, :])

    for i in range(cnt):  # 经过ransac筛出来的点进行H变换
        src_pts_left_moved[i][0] = M[0][0] * src_pts_left_[i][0] + M[0][1] * src_pts_left_[i][1] + M[0][2]
        src_pts_left_moved[i][1] = M[1][0] * src_pts_left_[i][0] + M[1][1] * src_pts_left_[i][1] + M[1][2]
        den = M[2][0] * src_pts_left_[i][0] + M[2][1] * src_pts_left_[i][1] + M[2][2]
        src_pts_left_moved[i][0] = src_pts_left_moved[i][0] / den
        src_pts_left_moved[i][1] = src_pts_left_moved[i][1] / den
    src_pts_left_moved = np.array(src_pts_left_moved)

    matches1 = draw_ransac_point(img1, img2, src_pts_left_moved, dst_pts_right_moved, src_pts_left_)

    return src_pts_left, src_pts_left_moved, dst_pts_right_moved, matches1, cnt


def draw_ransac_point(src1, src2, src_after_pts, dst_pts, src_pts):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
    _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)
    # 左图绿色,右图红色
    residual = 0
    for i in range(len(src_after_pts)):
        x1 = int(src_after_pts[i][0] + src1.shape[1])
        y1 = int(src_after_pts[i][1])
        cv2.circle(output, (x1, y1), 10, (0,0,255), -1)
        x2 = int(dst_pts[i][0] + src1.shape[1])
        y2 = int(dst_pts[i][1])
        cv2.circle(output, (x2, y2), 10, (0,165,255), -1)
        cv2.line(output,  (x1, y1), (x2, y2), (255, 255, 255), 1)
        x3 = int(src_pts[i][0])
        y3 = int(src_pts[i][1])
        cv2.circle(output, (x3, y3), 10, (0,0,255), -1)
        residual = float((src_after_pts[i][0] - dst_pts[i][0]) * (src_after_pts[i][0] - dst_pts[i][0]) + (
                src_after_pts[i][1] - dst_pts[i][1]) * (src_after_pts[i][1] - dst_pts[i][1])) + residual
    all.Ransac_residuals = residual / len(src_after_pts)
    return output
