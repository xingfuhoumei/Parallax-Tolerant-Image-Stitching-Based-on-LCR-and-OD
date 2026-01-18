# goal:实现能量函数的功能
import cv2
import numpy as np
from numba import jit
from sklearn import preprocessing


def get_cost(result_img1, result_img2, dst, dst_min_overlap_point_x, src_after,new,flag):

    matches_scores = get_matches_scores(dst, src_after)
    dst[:, 0] = dst[:, 0] - dst_min_overlap_point_x
    h, w = result_img1.shape[0], result_img1.shape[1]
    distances = np.zeros((h, w), dtype=np.float64)
    dis_tmp = np.zeros((h, w), dtype=np.float64)
    gama = 0.01
    sigma = 0.05 * min(h, w)
    # 得到差分图
    edge_map1 = cv2.subtract(result_img1, result_img2)  # subtract the intersection, leaving just the new part to union
    edge_map2 = cv2.subtract(result_img2, result_img1)
    # # cv2.imshow("edge_map1", edge_map1)
    # # cv2.imshow("edge_map2", edge_map2)
    # cv2.waitKey()
    edge = edge_map1 + edge_map2
    diff_weight = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
    # ret, diff_weight = cv2.threshold(diff_weight, 0, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # diff_weight = cv2.morphologyEx(diff_weight, cv2.MORPH_CLOSE, kernel)
    # # cv2.imshow("diff_weight1", diff_weight)
    # # # cv2.imshow("diff_weight",diff_weight)
    # # cv2.waitKey()
    # # 不超出边缘（水平拼接）
    points_weight = get_point_weight(dst, h, w, dis_tmp, distances, matches_scores)
    Gaussian_weight = cv2.GaussianBlur(points_weight, (3, 3), sigmaX=0)
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    normalize_Gaussian_weight = minmax_scaler.fit_transform(Gaussian_weight)
    normalize_Gaussian_weight1 = np.array(normalize_Gaussian_weight, dtype='uint8')
    # # cv2.imshow("normalize_Gaussian_weight1", normalize_Gaussian_weight1)
    # normalize_diff_weight = minmax_scaler.fit_transform(diff_weight)
    # normalize_diff_weight1 = np.array(normalize_diff_weight, dtype='uint8')
    # # # cv2.imshow("normalize_diff_weight1", normalize_diff_weight1)
    # # cv2.waitKey()
    if flag:
        img_parallax1 = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        weight = 0.6 * diff_weight + 0.1 * normalize_Gaussian_weight + 0.3 * img_parallax1
        # weight = 0.3 * diff_weight + 0.4 * normalize_Gaussian_weight + 0.3 * img_parallax1
        weight = 0.2 * diff_weight + 0.8 * normalize_Gaussian_weight + 0 * img_parallax1

    else:
        weight = 0.4 * diff_weight + 0.6 * normalize_Gaussian_weight

    weight[:, 0] = 255
    weight[:, -1] = 255
    weight1 = np.array(weight, dtype='uint8')
    # cv2.imshow("weight1",weight1)
    return weight, weight1



@jit(nopython=True)
def get_matches_scores(dst, src_after):
    score = np.zeros((1, dst.shape[0]))
    for i in range(dst.shape[0]):
        score[0][i] = pow((pow((dst[i][0] - src_after[i][0]), 2) + pow((dst[i][1] - src_after[i][1]), 2)), 0.5)
    return score


# 利用numba进行加速
@jit(nopython=True)
def get_point_weight(dst, h, w, dis_tmp, distances, matches_scores):
    for i in range(dst.shape[0]):
        x, y = dst[i][0], dst[i][1]
        # 行遍历 得到了高斯权重
        for row in range(h):
            # 列遍历
            for col in range(w):
                dis = matches_scores[0][i] * (np.sqrt((x - col) ** 2 + (y - row) ** 2))
                dis_tmp[row][col] = +dis
        distances = distances + dis_tmp
    return distances


