import math
from random import random

import cv2
import nudged
import numpy as np
from numpy import linalg
import all


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def get_true_points(img1, img2, src_pts_RT_8, dst_pts_RT_8, good_num_RT_8, scores_RT8):
    # 初始化+
    size_left = Size(img1.shape[1], img1.shape[0])
    size_right = Size(img2.shape[1], img2.shape[0])
    size_RT = Size(2, 2)
    size_A = Size(3, 3)
    size_H = Size(4, 4)
    size_H1 = Size(5, 5)
    size_H2 = Size(6, 6)
    size_H3 = Size(7, 7)
    size_H4 = Size(7, 7)

    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_pts_RT_8)
    # 0.8的匹配对进行分块 右图的块号
    dst_pts_grid_idx_RT_8 = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_pts_RT_8,
                                             size_RT)
    # 做RT变换返回平均点计算得到的RT矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    trans_8, RT_8, src_pts_after_RT8, RT_left_center_8, RT_right_center_8, \
    list_left_center_8, center_src_grid_idx_8 = get_RT_transform(
        src_pts_RT_8, dst_pts_RT_8, dst_pts_grid_idx_RT_8)
    residuals = get_adv_residual(src_pts_after_RT8, dst_pts_RT_8)
    # 寻找真正的错误匹配对
    real_index, false_index = reject_RT_point(
        dst_pts_grid_idx_RT_8,
        dst_pts_RT_8, first_point, X_Axis_vector,
        Y_Axis_vector, size_RT, size_right, src_pts_after_RT8, xmin, xmax, ymin, ymax)
    # 变换前左右图剔除后的点
    src_real_RT8 = src_pts_RT_8[real_index]
    dst_real_RT8 = dst_pts_RT_8[real_index]

    # 变换后剔除后的点
    src_after_real_RT8 = src_pts_after_RT8[real_index]
    dst_after_real_RT8 = dst_pts_RT_8[real_index]
    all.RT_residuals = get_adv_residual(src_after_real_RT8, dst_after_real_RT8)
    # 左图原始被剔除的点
    src_false_RT8 = src_pts_RT_8[false_index]
    # 右图原始被剔除的点
    dst_false_RT8 = dst_pts_RT_8[false_index]
    # # ##########       A变换        ##########
    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_real_RT8)
    dst_pts_grid_idx_A = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_real_RT8, size_A)
    # 做A变换返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    A, src_pts_after_A, A_left_center, A_right_center, list_left_center, center_src_grid_idx = get_A_transform(
        src_real_RT8, dst_real_RT8, dst_pts_grid_idx_A)
    # 寻找真正的错误匹配对
    real_index, false_index = reject_A_point(dst_pts_grid_idx_A, dst_real_RT8, first_point, X_Axis_vector,
                                             Y_Axis_vector, size_A, size_right, src_pts_after_A, xmin, xmax, ymin, ymax)
    src_real_A = src_real_RT8[real_index]
    dst_real_A = dst_real_RT8[real_index]
    # 变换后剔除后的点
    src_after_real_A = src_pts_after_A[real_index]
    dst_after_real_A = dst_real_RT8[real_index]
    all.A_residuals = get_adv_residual(src_after_real_A, dst_after_real_A)
    # # ##########       H变换        ##########
    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_real_A)
    dst_pts_grid_idx_H = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_real_A, size_H)
    # # 做A变换返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    H, src_pts_after_H, H_left_center, H_right_center, list_left_center, center_src_grid_idx = get_H_transform(
        src_real_A, dst_real_A, dst_pts_grid_idx_H, size_H)
    # # 寻找真正的错误匹配对
    real_index, false_index = reject_H_point(dst_pts_grid_idx_H, dst_real_A, first_point, X_Axis_vector,
                                             Y_Axis_vector, size_H, size_right, src_pts_after_H, xmin, xmax, ymin, ymax)
    # print(false_index)
    # # 变换前左右图剔除后的点
    src_real_H = src_real_A[real_index]
    dst_real_H = dst_real_A[real_index]
    src_after_real_H = src_pts_after_H[real_index]
    dst_after_real_H = dst_real_A[real_index]
    all.H_residuals = get_adv_residual(src_after_real_H, dst_after_real_H)
    # # ##########       H1变换        ##########
    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_real_H)
    dst_pts_grid_idx_H1 = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_real_H, size_H1)
    # # 做A变换返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    H1, src_pts_after_H1, H_left_center, H_right_center, list_left_center, center_src_grid_idx = get_H_transform(
        src_real_H, dst_real_H, dst_pts_grid_idx_H1, size_H1)
    # # 寻找真正的错误匹配对
    real_index, false_index = reject_H1_point(dst_pts_grid_idx_H1, dst_real_H, first_point, X_Axis_vector,
                                              Y_Axis_vector, size_H1, size_right, src_pts_after_H1, xmin, xmax, ymin,
                                              ymax)
    # # 变换前左右图剔除后的点
    src_real_H1 = src_real_H[real_index]
    dst_real_H1 = dst_real_H[real_index]
    # 变换后剔除后的点
    src_after_real_H1 = src_pts_after_H1[real_index]
    dst_after_real_H1 = dst_real_H[real_index]
    all.H1_residuals = get_adv_residual(src_after_real_H1, dst_after_real_H1)
    # # ##########       H2变换        ##########
    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_real_H1)
    dst_pts_grid_idx_H2 = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_real_H1, size_H2)
    # # 做A变换返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    H2, src_pts_after_H2, H_left_center, H_right_center, list_left_center, center_src_grid_idx = get_H_transform(
        src_real_H1, dst_real_H1, dst_pts_grid_idx_H2, size_H2)

    # # 寻找真正的错误匹配对
    real_index, false_index = reject_H2_point(dst_pts_grid_idx_H2, dst_real_H1, first_point, X_Axis_vector,
                                              Y_Axis_vector, size_H2, size_right, src_pts_after_H2, xmin, xmax, ymin,
                                              ymax)
    # # print(false_index)
    # # 变换前左右图剔除后的点
    src_real_H2 = src_real_H1[real_index]
    dst_real_H2 = dst_real_H1[real_index]
    # 变换后剔除后的点
    src_after_real_H2 = src_pts_after_H2[real_index]
    dst_after_real_H2 = dst_real_H1[real_index]
    all.H2_residuals = get_adv_residual(src_after_real_H2, dst_after_real_H2)
    # # ##########       H3变换        ##########
    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_real_H2)
    dst_pts_grid_idx_H3 = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_real_H2, size_H3)
    # # 做A变换返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    H3, src_pts_after_H3, H_left_center, H_right_center, list_left_center, center_src_grid_idx = get_H_transform(
        src_real_H2, dst_real_H2, dst_pts_grid_idx_H3, size_H3)
    # # 寻找真正的错误匹配对
    real_index, false_index = reject_H3_point(dst_pts_grid_idx_H3, dst_real_H2, first_point, X_Axis_vector,
                                              Y_Axis_vector, size_H3, size_right, src_pts_after_H3, xmin, xmax, ymin,
                                              ymax)
    # # print(false_index)
    # # 变换前左右图剔除后的点
    src_real_H3 = src_real_H2[real_index]
    dst_real_H3 = dst_real_H2[real_index]
    # 变换后剔除后的点
    src_after_real_H3 = src_pts_after_H3[real_index]
    dst_after_real_H3 = dst_real_H2[real_index]
    all.H3_residuals = get_adv_residual(src_after_real_H3, dst_after_real_H3)
    # # ##########       最后一次剔除        ##########
    real_index = []
    false_index = []
    for i in range(src_real_H3.shape[0]):
        if (pow((pow((src_after_real_H3[i][0] - dst_real_H3[i][0]), 2) + pow(
                (src_after_real_H3[i][1] - dst_real_H3[i][1]), 2)), 0.5) <= pow(
            (pow((xmax - xmin) / 15, 2) + pow((ymax - ymin) / 15, 2)), 0.5)):
            real_index.append(i)
        else:
            false_index.append(i)
    all.real_index_number = len(real_index)
    all.false_index_number = len(false_index)
    src_real_H4 = src_real_H3[real_index]
    dst_real_H4 = dst_real_H3[real_index]
    src_after_real_H4 = src_after_real_H3[real_index]
    all.final_residuals = get_adv_residual(src_after_real_H4, dst_real_H4)
    src_false_H4 = src_real_H3[false_index]
    # 右图原始被剔除的点
    dst_false_H4 = dst_real_H3[false_index]
    first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax = get_vector(dst_real_H4)
    dst_pts_grid_idx_H4 = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_real_H4, size_H4)

    # 做A变换返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,变换后中心序号center_grid_idx
    H4, src_pts_after_H4, H_left_center, H_right_center, list_left_center, center_src_grid_idx = get_H_transform(
        src_real_H4, dst_real_H4, dst_pts_grid_idx_H4, size_H4)
    # # 输入: src_pts变换前的点, src_pts2变换后的点, dst_pts2右图点
    # # 输入:图1,图2,左图原始特征点,变换后左图特征点,右图特征点,左图中心点,右图中心点,左图变换后的中心点
    # # 左图原始特征点:绿色, 左图变换后特征点:黑色,右原始特征点:红色
    # # 左图中心点:黄色 右图中心点:蓝色 变换后左图中心点:黑色
    H4matches = draw_point(img1, img2, src_real_H4, src_pts_after_H4, dst_real_H4, H_left_center,
                           H_right_center,
                           list_left_center, size_H4, xmin, xmax, ymin, ymax)
    return src_real_H4, dst_real_H4, H4, src_after_real_H4, H4matches


def get_really_index(first_point, X_Axis_vector, Y_Axis_vector, dst_pts, size_RT):
    x_length = pow((pow(X_Axis_vector[0], 2) + pow(X_Axis_vector[1], 2)), 0.5) + 0.1
    y_length = pow((pow(Y_Axis_vector[0], 2) + pow(Y_Axis_vector[1], 2)), 0.5) + 0.1
    first_point = first_point.reshape(1, 2)
    x0 = first_point[0][0]
    y0 = first_point[0][1]
    normalize_points = np.zeros(dst_pts.shape)
    input_points_index = np.zeros(dst_pts.shape[0])
    for i in range(dst_pts.shape[0]):
        # 真实向量
        x1 = dst_pts[i][0] - x0
        y1 = dst_pts[i][1] - y0
        vector = np.array([x1, y1])
        # 查断点调试，可注释a，b两行
        a = (np.dot(vector, X_Axis_vector) / np.linalg.norm(X_Axis_vector)) / x_length
        b = (np.dot(vector, Y_Axis_vector) / np.linalg.norm(Y_Axis_vector)) / y_length
        normalize_points[i][0] = (np.dot(vector, X_Axis_vector) / np.linalg.norm(X_Axis_vector)) / x_length
        normalize_points[i][1] = (np.dot(vector, Y_Axis_vector) / np.linalg.norm(Y_Axis_vector)) / y_length
        x_divided = int(math.floor(normalize_points[i][0] * size_RT.width))
        y_divided = int(math.floor(normalize_points[i][1] * size_RT.height))
        input_points_index[i] = x_divided + y_divided * size_RT.width
    return input_points_index


def get_vector(dst_pts):
    xmin = dst_pts[np.argmin(dst_pts[:, 0]), 0]
    xmax = dst_pts[np.argmax(dst_pts[:, 0]), 0]
    ymin = dst_pts[np.argmin(dst_pts[:, 1]), 1]
    ymax = dst_pts[np.argmax(dst_pts[:, 1]), 1]
    first_point = np.array([xmin, ymin])
    X_Axis_vector = np.array([xmax - xmin, 0])
    Y_Axis_vector = np.array([0, ymax - ymin])
    return first_point, X_Axis_vector, Y_Axis_vector, xmin, xmax, ymin, ymax


# RT变换
def get_RT_transform(src_pts, dst_pts, dst_pts_grid_idx_RT_8):
    idx1 = 4
    seed_left = np.zeros((idx1, 2))
    seed_right = np.zeros((idx1, 2))
    list_left = []
    list_right = []
    center_grid_idx = []
    # i = 0
    for i in range(idx1):
        sum_x_left = 0
        sum_y_left = 0
        sum_x_right = 0
        sum_y_right = 0
        center_x_left = 0
        center_y_left = 0
        center_x_right = 0
        center_y_right = 0
        cnt = 0
        for j in range(len(dst_pts_grid_idx_RT_8)):
            if dst_pts_grid_idx_RT_8[j] == i:
                cnt += 1
                sum_x_left += src_pts[j][0]
                sum_y_left += src_pts[j][1]
                sum_x_right += dst_pts[j][0]
                sum_y_right += dst_pts[j][1]
        if cnt != 0:
            center_x_left = float(sum_x_left / cnt)
            center_y_left = float(sum_y_left / cnt)
            center_x_right = float(sum_x_right / cnt)
            center_y_right = float(sum_y_right / cnt)
            seed_left[i][0] = center_x_left
            seed_left[i][1] = center_y_left
            seed_right[i][0] = center_x_right
            seed_right[i][1] = center_y_right
            # 保留左图变换前每个点的索引
            center_grid_idx.append(i)
            list_left.append(seed_left[i])
            list_right.append(seed_right[i])
        # i += 1
    list_left = np.array(list_left)
    list_right = np.array(list_right)

    # 得到4对种子点seed_left seed_right 并以此计算RT
    # 将左图的所有特征点src_pts做RT变换
    # 把src_pts转化为list格式的test
    test = []
    for element in src_pts:
        test.append(list(element))
    # 左边的原点 变换到右边的原点
    trans = nudged.estimate(list_left, list_right)
    output = trans.transform(test)
    RT = np.array(trans.get_matrix())
    # 将左图的所有特征点src_pts做RT变换
    src_pts1 = np.zeros(src_pts.shape)
    list_left_center = np.zeros(list_left.shape)
    for i in range(len(dst_pts_grid_idx_RT_8)):
        src_pts1[i][0] = trans.s * src_pts[i][0] - trans.r * src_pts[i][1] + trans.tx
        src_pts1[i][1] = trans.r * src_pts[i][0] + trans.s * src_pts[i][1] + trans.ty
        # RT左图中心点变换后
    for i in range(len(list_left)):
        list_left_center[i][0] = trans.s * list_left[i][0] - trans.r * list_left[i][1] + trans.tx
        list_left_center[i][1] = trans.r * list_left[i][0] + trans.s * list_left[i][1] + trans.ty
        # 返回平均点计算得到的RT矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,center_grid_idx
    return trans, RT, src_pts1, list_left, list_right, list_left_center, center_grid_idx


# A变换
def get_A_transform(src_pts, dst_pts, dst_pts_grid_idx_A):
    idx2 = 3 + 3 * (3 - 1)
    seed_left = np.zeros((idx2, 2))
    seed_right = np.zeros((idx2, 2))
    list_left = []
    list_right = []
    center_grid_idx = []

    # i = 0
    for i in range(idx2):
        sum_x_left = 0
        sum_y_left = 0
        sum_x_right = 0
        sum_y_right = 0
        cnt = 0
        for j in range(len(dst_pts_grid_idx_A)):
            if dst_pts_grid_idx_A[j] == i:
                cnt += 1
                sum_x_left += src_pts[j][0]
                sum_y_left += src_pts[j][1]
                sum_x_right += dst_pts[j][0]
                sum_y_right += dst_pts[j][1]
        if cnt != 0:
            center_x_left = float(sum_x_left / cnt)
            center_y_left = float(sum_y_left / cnt)
            center_x_right = float(sum_x_right / cnt)
            center_y_right = float(sum_y_right / cnt)
            seed_left[i][0] = center_x_left
            seed_left[i][1] = center_y_left
            seed_right[i][0] = center_x_right
            seed_right[i][1] = center_y_right
            center_grid_idx.append(i)
            list_left.append(seed_left[i])
            list_right.append(seed_right[i])
    list_left = np.array(list_left)
    list_right = np.array(list_right)
    # 得到<=9对的种子点list_left list_right 并以此计算A
    Affine_matrix, _ = np.array(cv2.estimateAffine2D(list_left, list_right, method=cv2.LMEDS))
    # print(Affine_matrix)
    # 将左图的所有特征点做A变换
    list_left = np.array(list_left)
    list_right = np.array(list_right)
    src_pts1 = np.zeros(src_pts.shape)
    left_center_after = np.zeros(list_left.shape)
    for i in range(len(dst_pts_grid_idx_A)):
        src_pts1[i][0] = Affine_matrix[0][0] * src_pts[i][0] + Affine_matrix[0][1] * src_pts[i][1] + Affine_matrix[0][2]
        src_pts1[i][1] = Affine_matrix[1][0] * src_pts[i][0] + Affine_matrix[1][1] * src_pts[i][1] + Affine_matrix[1][2]
    # A = np.row_stack((Affine_matrix, [0, 0, 1]))
    # print
    # 左图中心点变换后
    for i in range(len(list_left)):
        left_center_after[i][0] = Affine_matrix[0][0] * list_left[i][0] + Affine_matrix[0][1] * list_left[i][1] + \
                                  Affine_matrix[0][2]
        left_center_after[i][1] = Affine_matrix[1][0] * list_left[i][0] + Affine_matrix[1][1] * list_left[i][1] + \
                                  Affine_matrix[1][2]
        # 返回平均点计算得到的A矩阵,变换后的左图点src_pts1,右图原点dst_pts,变换前作图中心点list_left,变换后左图中心点list_left_center,center_grid_idx
    return Affine_matrix, src_pts1, list_left, list_right, left_center_after, center_grid_idx


# H变换
def get_H_transform(src_pts, dst_pts, src_pts_grid_idx, size):
    idx3 = size.width * size.height
    seed_left = np.zeros((idx3, 2))
    seed_right = np.zeros((idx3, 2))
    list_left = []
    list_right = []
    center_grid_idx = []
    # i = 0 i是序号块数量
    for i in range(idx3):
        sum_x_left = 0
        sum_y_left = 0
        sum_x_right = 0
        sum_y_right = 0
        center_x_left = 0
        center_y_left = 0
        center_x_right = 0
        center_y_right = 0
        cnt = 0
        # j是左图原点的数量
        for j in range(len(src_pts_grid_idx)):
            if src_pts_grid_idx[j] == i:
                cnt += 1
                sum_x_left += src_pts[j][0]
                sum_y_left += src_pts[j][1]
                sum_x_right += dst_pts[j][0]
                sum_y_right += dst_pts[j][1]
        if cnt != 0:
            center_x_left = float(sum_x_left / cnt)
            center_y_left = float(sum_y_left / cnt)
            center_x_right = float(sum_x_right / cnt)
            center_y_right = float(sum_y_right / cnt)
            seed_left[i][0] = center_x_left
            seed_left[i][1] = center_y_left
            seed_right[i][0] = center_x_right
            seed_right[i][1] = center_y_right
            center_grid_idx.append(i)
            list_left.append(seed_left[i])
            list_right.append(seed_right[i])
        # i += 1
    H_matrix = np.zeros((seed_left.shape[0] * 2, 9))
    for i in range(H_matrix.shape[0]):
        # 奇数行
        if i % 2 == 0:
            n = int(i / 2)
            H_matrix[i][0] = seed_left[n][0]
            H_matrix[i][1] = seed_left[n][1]
            H_matrix[i][2] = 1
            H_matrix[i][6] = -seed_right[n][0] * seed_left[n][0]
            H_matrix[i][7] = -seed_right[n][0] * seed_left[n][1]
            H_matrix[i][8] = -seed_right[n][0]
        # 偶数行
        elif i % 2 == 1:
            m = int(i / 2)
            H_matrix[i][3] = seed_left[m][0]
            H_matrix[i][4] = seed_left[m][1]
            H_matrix[i][5] = 1
            H_matrix[i][6] = -seed_right[m][1] * seed_left[m][0]
            H_matrix[i][7] = -seed_right[m][1] * seed_left[m][1]
            H_matrix[i][8] = -seed_right[m][1]
    U, sigma, VT = linalg.svd(H_matrix)
    H = np.array(VT)
    # VT最后一行的行向量即为最小特征值对应的特征向量
    k1 = 1 / VT[8, 8]
    Homegraphy = k1 * H[8, :].reshape(3, 3)
    # print('SVD矩阵', Homegraphy)
    list_left = np.array(list_left)
    list_right = np.array(list_right)
    # 得到<=1145组种子点list_left list_right 并以此计算H
    # 采用svd分解进行加速计算H
    Homegraphy1, _ = np.array(cv2.findHomography(list_left, list_right))
    # print('findhomography', Homegraphy1)
    src_pts_ = np.zeros(src_pts.shape)
    left_center_after = np.zeros(list_left.shape)
    # 做H变换
    for i in range(len(src_pts_grid_idx)):
        src_pts_[i][0] = Homegraphy[0][0] * src_pts[i][0] + Homegraphy[0][1] * src_pts[i][1] + Homegraphy[0][2]
        src_pts_[i][1] = Homegraphy[1][0] * src_pts[i][0] + Homegraphy[1][1] * src_pts[i][1] + Homegraphy[1][2]
        den = Homegraphy[2][0] * src_pts[i][0] + Homegraphy[2][1] * src_pts[i][1] + Homegraphy[2][2]
        src_pts_[i][0] = src_pts_[i][0] / den
        src_pts_[i][1] = src_pts_[i][1] / den
    # 中心点H变换
    for i in range(len(list_left)):
        left_center_after[i][0] = Homegraphy[0][0] * list_left[i][0] + Homegraphy[0][1] * list_left[i][1] + \
                                  Homegraphy[0][2]
        left_center_after[i][1] = Homegraphy[1][0] * list_left[i][0] + Homegraphy[1][1] * list_left[i][1] + \
                                  Homegraphy[1][2]
        den = Homegraphy[2][0] * list_left[i][0] + Homegraphy[2][1] * list_left[i][1] + Homegraphy[2][2]
        left_center_after[i][0] = left_center_after[i][0] / den
        left_center_after[i][1] = left_center_after[i][1] / den
    return Homegraphy, src_pts_, list_left, list_right, left_center_after, center_grid_idx


def get_adv_residual(src_real_RT8, dst_real_RT8):
    residual = 0
    for i in range(src_real_RT8.shape[0]):
        residual = pow(
            (pow((src_real_RT8[i][0] - dst_real_RT8[i][0]), 2) + pow((src_real_RT8[i][1] - dst_real_RT8[i][1]), 2)),
            0.5) + residual
    residuals = residual / src_real_RT8.shape[0]
    return residuals


# 右图序号，右图原点，变换矩阵，左图原点，左图x轴，左图y轴，rt_size，左图left
def reject_RT_point(dst_pts_grid_idx_RT_8, dst_pts, first_point, X_Axis_vector, Y_Axis_vector, size_RT,
                    size_right,
                    src_pts_after_RT8, xmin, xmax, ymin, ymax):
    next_start_vector = first_point
    X_Axis_vector = X_Axis_vector
    Y_Axis_vector = Y_Axis_vector
    # 中线x的坐标
    mid_width = xmin + (xmax - xmin) / 2
    # x的长度
    offset_width = xmax - xmin
    # 中线y的坐标
    mid_height = ymin + (ymax - ymin) / 2
    # y的长度
    offset_height = ymax - ymin
    real_index = []
    false_index = []
    src_pts_index = get_really_index(next_start_vector, X_Axis_vector, Y_Axis_vector, src_pts_after_RT8, size_RT)

    residual = 0
    if len(src_pts_index) == len(dst_pts_grid_idx_RT_8):
        for i in range(len(src_pts_index)):
            if (dst_pts_grid_idx_RT_8[i] == src_pts_index[i] and size_right.width >= src_pts_after_RT8[i][0] >= 0 and
                0 <= src_pts_after_RT8[i][1] <= size_right.height) \
                    or mid_width - offset_width * 1 / 15 < dst_pts[i][0] < mid_width + offset_width * 1 / 15 \
                    or mid_height - offset_height * 1 / 15 < dst_pts[i][1] < mid_height + offset_height * 1 / 15:
                real_index.append(i)
            else:
                false_index.append(i)
    PrintRt = "一共"+str(len(src_pts_index))+"个点,剔除了"+str(len(false_index))+"个点,保留了"+str(len(real_index))+"个点"
    print("一共", len(src_pts_index), "个点", "剔除了", len(false_index), "个点", "保留了", len(real_index), "个点")
    return real_index, false_index


# 左图序号，右图原点，变换矩阵，左图原点，左图x轴，左图y轴，rt_size，左图left
def reject_A_point(dst_pts_grid_idx_A, dst_real_RT8, first_point, X_Axis_vector, Y_Axis_vector, size_A, size_right,
                   src_pts_after_A, xmin, xmax, ymin, ymax):
    real_index = []
    false_index = []
    # 中线x的坐标
    mid_width = xmax - xmin
    # x的长度
    offset_width = (xmax - xmin) * 1 / 15
    # 中线y的坐标
    mid_height = ymax - ymin
    # y的长度
    offset_height = (ymax - ymin) * 1 / 15
    src_pts_index = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, src_pts_after_A, size_A)
    if len(src_pts_index) == len(dst_pts_grid_idx_A):
        for i in range(len(src_pts_index)):
            # if dst_pts_index[i]==src_pts_index[i] or size_right.width/size_RT.width-50<src_pts_RT_8[i][0]<size_right.width/size_RT.width+50 or size_right.height/size_RT.width-50<src_pts_RT_8[i][1]<size_right.height/size_RT.width+50:
            if (dst_pts_grid_idx_A[i] == src_pts_index[i] and size_right.width >= src_pts_after_A[i][0] >= 0 and 0 <=
                src_pts_after_A[i][1] <= size_right.height) \
                    or mid_width / 3 + xmin - offset_width < dst_real_RT8[i][
                0] < mid_width / 3 + xmin + offset_width \
                    or mid_height / 3 + ymin - offset_height < dst_real_RT8[i][
                1] < mid_height / 3 + ymin + offset_height \
                    or mid_width * 2 / 3 + xmin - offset_width < dst_real_RT8[i][
                0] < mid_width * 2 / 3 + xmin + offset_width \
                    or mid_height * 2 / 3 + ymin - offset_height < dst_real_RT8[i][
                1] < mid_height * 2 / 3 + ymin + offset_height:
                real_index.append(i)
            else:
                false_index.append(i)
    print("一共", len(src_pts_index), "个点", "剔除了", len(false_index), "个点", "保留了", len(real_index), "个点")
    # print("A的错误点序号", false_index)
    return real_index, false_index


def reject_H_point(dst_pts_grid_idx_H, dst_real_A, first_point, X_Axis_vector,
                   Y_Axis_vector, size_H, size_right, src_pts_after_H, xmin, xmax, ymin, ymax):
    real_index = []
    false_index = []
    # 中线x的坐标
    mid_width = xmax - xmin
    # x的长度
    offset_width = (xmax - xmin) * 1 / 15
    # 中线y的坐标
    mid_height = ymax - ymin
    # y的长度
    offset_height = (ymax - ymin) * 1 / 15
    src_pts_index = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, src_pts_after_H,
                                     size_H)
    if len(src_pts_index) == len(dst_pts_grid_idx_H):
        for i in range(len(dst_pts_grid_idx_H)):
            if (dst_pts_grid_idx_H[i] == src_pts_index[i]
                and size_right.width >= src_pts_after_H[i][0] >= 0
                and size_right.height >= src_pts_after_H[i][1] >= 0) \
                    or mid_width / 4 + xmin - offset_width < dst_real_A[i][
                0] < mid_width / 4 + xmin + offset_width \
                    or mid_height / 4 + ymin - offset_height < dst_real_A[i][
                1] < mid_height / 4 + ymin + offset_height \
                    or mid_width * 2 / 4 + xmin - offset_width < dst_real_A[i][
                0] < mid_width * 2 / 4 + xmin + offset_width \
                    or mid_height * 2 / 4 + ymin - offset_height < dst_real_A[i][
                1] < mid_height * 2 / 4 + ymin + offset_height \
                    or mid_width * 3 / 4 + xmin - offset_width < dst_real_A[i][
                0] < mid_width * 3 / 4 + xmin + offset_width \
                    or mid_height * 3 / 4 + ymin - offset_height < dst_real_A[i][
                1] < mid_height * 3 / 4 + ymin + offset_height:
                real_index.append(i)
            else:
                false_index.append(i)
    print("一共", len(src_pts_index), "个点", "剔除了", len(false_index), "个点", "保留了", len(real_index), "个点")
    return real_index, false_index


def reject_H1_point(dst_pts_grid_idx_H, dst_real_A, first_point, X_Axis_vector,
                    Y_Axis_vector, size_H1, size_right, src_pts_after_H, xmin, xmax, ymin, ymax):
    real_index = []
    false_index = []
    src_pts_index = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, src_pts_after_H,
                                     size_H1)
    if len(src_pts_index) == len(dst_pts_grid_idx_H):
        for i in range(len(dst_pts_grid_idx_H)):
            if (dst_pts_grid_idx_H[i] == src_pts_index[i]
                    and size_right.width >= src_pts_after_H[i][0] >= 0
                    and size_right.height >= src_pts_after_H[i][1] >= 0):
                #     or xmin + (mid_width / 5) - offset_width < dst_real_A[i][
                # 0] < xmin + (mid_width / 5) + offset_width \
                #     or ymin + (mid_height / 5) - offset_height < dst_real_A[i][
                # 1] < ymin + (mid_height / 5) + offset_height \
                #     or xmin + (mid_width * 2 / 5) - offset_width < dst_real_A[i][
                # 0] < xmin + (mid_width * 2 / 5) + offset_width \
                #     or ymin + (mid_height * 2 / 5) - offset_height < dst_real_A[i][
                # 1] < ymin + (mid_height * 2 / 5) + offset_height \
                #     or xmin + (mid_width * 3 / 5) - offset_width < dst_real_A[i][
                # 0] < xmin + (mid_width * 3 / 5) + offset_width \
                #     or ymin + (mid_height * 3 / 5) - offset_height < dst_real_A[i][
                # 1] < ymin + (mid_height * 3 / 5) + offset_height \
                #     or xmin + (mid_width * 4 / 5) - offset_width < dst_real_A[i][
                # 0] < xmin + (mid_width * 4 / 5) + offset_width \
                #     or ymin + (mid_height * 4 / 5) - offset_height < dst_real_A[i][
                # 1] < ymin + (mid_height * 4 / 5) + offset_height:
                real_index.append(i)
            else:
                false_index.append(i)
    print("一共", len(src_pts_index), "个点", "剔除了", len(false_index), "个点", "保留了", len(real_index), "个点")
    return real_index, false_index


def reject_H2_point(dst_pts_grid_idx_H, dst_real_A, first_point, X_Axis_vector,
                    Y_Axis_vector, size_H2, size_right, src_pts_after_H, xmin, xmax, ymin,
                    ymax):
    real_index = []
    false_index = []
    # 中线x的坐标
    mid_width = xmax - xmin
    # x的长度
    offset_width = (xmax - xmin) * 1 / 15
    # 中线y的坐标
    mid_height = ymax - ymin
    # y的长度
    offset_height = (ymax - ymin) * 1 / 15
    src_pts_index = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, src_pts_after_H,
                                     size_H2)
    if len(src_pts_index) == len(dst_pts_grid_idx_H):
        for i in range(len(dst_pts_grid_idx_H)):
            if (dst_pts_grid_idx_H[i] == src_pts_index[i]
                and size_right.width >= src_pts_after_H[i][0] >= 0
                and size_right.height >= src_pts_after_H[i][1] >= 0) \
                    or xmin + mid_width / 6 - offset_width < dst_real_A[i][
                0] < xmin + mid_width / 6 + offset_width \
                    or ymin + mid_height / 6 - offset_height < dst_real_A[i][
                1] < ymin + mid_height / 6 + offset_height \
                    or xmin + mid_width * 2 / 6 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 2 / 6 + offset_width \
                    or ymin + mid_height * 2 / 6 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 2 / 6 + offset_height \
                    or xmin + mid_width * 3 / 6 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 3 / 6 + offset_width \
                    or ymin + mid_height * 3 / 6 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 3 / 6 + offset_height \
                    or xmin + mid_width * 4 / 6 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 4 / 6 + offset_width \
                    or ymin + mid_height * 4 / 6 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 4 / 6 + offset_height \
                    or xmin + mid_width * 5 / 6 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 5 / 6 + offset_width \
                    or ymin + mid_height * 5 / 6 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 5 / 6 + offset_height \
                    :
                real_index.append(i)
            else:
                false_index.append(i)
    print("一共", len(src_pts_index), "个点", "剔除了", len(false_index), "个点", "保留了", len(real_index), "个点")
    return real_index, false_index


def reject_H3_point(dst_pts_grid_idx_H, dst_real_A, first_point, X_Axis_vector,
                    Y_Axis_vector, size_H3, size_right, src_pts_after_H, xmin, xmax, ymin,
                    ymax):
    real_index = []
    false_index = []
    # 中线x的坐标
    mid_width = xmax - xmin
    # x的长度
    offset_width = (xmax - xmin) * 1 / 15
    # 中线y的坐标
    mid_height = ymax - ymin
    # y的长度
    offset_height = (ymax - ymin) * 1 / 15
    src_pts_index = get_really_index(first_point, X_Axis_vector, Y_Axis_vector, src_pts_after_H,
                                     size_H3)
    if len(src_pts_index) == len(dst_pts_grid_idx_H):
        for i in range(len(dst_pts_grid_idx_H)):
            if (dst_pts_grid_idx_H[i] == src_pts_index[i]
                and size_right.width >= src_pts_after_H[i][0] >= 0
                and size_right.height >= src_pts_after_H[i][1] >= 0) \
                    or xmin + mid_width / 7 - offset_width < dst_real_A[i][
                0] < xmin + mid_width / 7 + offset_width \
                    or ymin + mid_height / 7 - offset_height < dst_real_A[i][
                1] < ymin + mid_height / 7 + offset_height \
                    or xmin + mid_width * 2 / 7 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 2 / 7 + offset_width \
                    or ymin + mid_height * 2 / 7 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 2 / 7 + offset_height \
                    or xmin + mid_width * 3 / 7 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 3 / 7 + offset_width \
                    or ymin + mid_height * 3 / 7 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 3 / 7 + offset_height \
                    or xmin + mid_width * 4 / 7 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 4 / 7 + offset_width \
                    or ymin + mid_height * 4 / 7 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 4 / 7 + offset_height \
                    or xmin + mid_width * 5 / 7 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 5 / 7 + offset_width \
                    or ymin + mid_height * 5 / 7 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 5 / 7 + offset_height \
                    or xmin + mid_width * 6 / 7 - offset_width < dst_real_A[i][
                0] < xmin + mid_width * 6 / 7 + offset_width \
                    or ymin + mid_height * 6 / 7 - offset_height < dst_real_A[i][
                1] < ymin + mid_height * 6 / 7 + offset_height \
                    :
                real_index.append(i)
            else:
                false_index.append(i)
    print("一共", len(src_pts_index), "个点", "剔除了", len(false_index), "个点", "保留了", len(real_index), "个点")
    return real_index, false_index


# 匹配点
# print("RT最初平均残差为", residuals)
# 输入: src_pts变换前的点, src_pts2变换后的点, dst_pts2右图点
# 输入:图1,图2,左图原始特征点,变换后左图特征点,右图特征点,左图中心点,右图中心点,左图变换后的中心点
# 左图原始特征点:绿色, 左图变换后特征点:黑色,右图原始特征点:红色
# 左图中心点:黄色 右图中心点:蓝色 变换后左图中心点:黑色
def draw_point(src1, src2, src_pts, src_pts2, dst_pts2, RT_left_center, RT_right_center, RT_after_left_center, size,
               xmin, xmax, ymin, ymax):
    size = size.height * size.width
    x7 = 0
    y7 = 0
    line = int(math.sqrt(size) - 1)
    width_left = src1.shape[1]
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    _1_255 = np.expand_dims(np.array(range(0, 25145), dtype='uint8'), 1)
    _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)
    # 左图绿色,右图红色
    residual = 0
    # 描点
    for i in range(len(src_pts)):
        # (x1,y1)左图点变换前的点,   (x2,y2)左图点变换后的点 , (x3,y3)右图原点
        x1 = int(src_pts[i][0])
        y1 = int(src_pts[i][1])
        x2 = int(src_pts2[i][0] + src1.shape[1])
        y2 = int(src_pts2[i][1])
        x3 = int(dst_pts2[i][0] + src1.shape[1])
        y3 = int(dst_pts2[i][1])
        # mei
        cv2.circle(output, (x1, y1), 10, (0,0,255), -1)
        cv2.circle(output, (x2, y2), 10, (0,0,255), -1)
        cv2.circle(output, (x3, y3), 10, (0,165,255), -1)
        cv2.line(output, (x2, y2), (x3, y3), (255, 255, 255), 1)
    residual = residual / len(src_pts)
    #
    # for i in range(len(RT_left_center)):
    #     x5 = int(RT_right_center[i][0] + src1.shape[1])
    #     y5 = int(RT_right_center[i][1])
    #     # 左图
    #     cv2.circle(output, (x5, y5), 4, (0,0,255), -1)
    # for i in range(len(RT_after_left_center)):
    #     x145 = int(RT_after_left_center[i][0] + src1.shape[1])
    #     y145 = int(RT_after_left_center[i][1])
    #     cv2.circle(output, (x145, y145), 4, (0, 215, 211), -1)
    # print('剃点前平均残差为', residual)

    # 网格线已被删除 - 只保留特征点和连线的绘制
    # cv2.line(output, (0, 0), (511, 511), (0,0,255), 5)  # 删除对角线
    # width_left = int(xmax - xmin)
    # height = int(ymax - ymin)
    # x7 = xmin
    # y7 = ymin
    # for i in range(line):
    #     x7 = int(width_left / math.sqrt(size) + x7)
    #     cv2.line(output, (x7 + src1.shape[1], ymin), (x7 + src1.shape[1], ymax), (0, 0, 0), 2)  # 删除垂直网格线
    #     y7 = int(height / math.sqrt(size) + y7)
    #     cv2.line(output, (int(src1.shape[1] + xmin), y7), (int(src1.shape[1] + xmax), y7), (0, 0, 0), 2)  # 删除水平网格线

    # cv2.line(output, (int(xmin + src1.shape[1]), int(ymin)), (int(xmax + src1.shape[1]), int(ymin)), (0, 0, 0), 2)  # 删除上边框
    # cv2.line(output, (int(xmin + src1.shape[1]), int(ymin)), (int(xmin + src1.shape[1]), int(ymax)), (0, 0, 0), 2)  # 删除左边框
    # cv2.line(output, (int(xmin + src1.shape[1]), int(ymax)), (int(xmax + src1.shape[1]), int(ymax)), (0, 0, 0), 2)  # 删除下边框
    # cv2.line(output, (int(xmax + src1.shape[1]), int(ymin)), (int(xmax + src1.shape[1]), int(ymax)), (0, 0, 0), 2)  # 删除右边框
    return output
