# -*- coding: utf-8 -*-
"""
RANSAC 特征匹配模块
===========
功能说明：
1. 使用 RANSAC 算法计算单应性矩阵
2. 过滤离群点，保留内点
3. 可视化 RANSAC 匹配结果

作者：gyd
日期：2023
"""

from typing import Tuple, Optional  # 类型注解

import cv2  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库

# 导入全局统计变量
import all


# ============== RANSAC 参数配置 ==============

# RANSAC 重投影误差阈值
# 较小的值会产生更严格的过滤
RANSAC_REPROJ_THRESHOLD = 4.0


# ============== 主要接口函数 ==============

def get_RANSAC_points(
    img1: np.ndarray,
    img2: np.ndarray,
    src_pts_RT_8: np.ndarray,
    dst_pts_RT_8: np.ndarray,
    good_num_RT_8: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    使用 RANSAC 算法获取内点匹配

    该函数通过 RANSAC 算法计算单应性矩阵，并过滤掉离群点

    参数:
        img1: 第一张输入图像
        img2: 第二张输入图像
        src_pts_RT_8: 源图像中的匹配点坐标
        dst_pts_RT_8: 目标图像中的匹配点坐标
        good_num_RT_8: 初始匹配点数量

    返回:
        src_pts_left: 源图像内点坐标
        src_pts_left_moved: 变换后的源图像内点坐标
        dst_pts_right_moved: 目标图像内点坐标
        matches1: RANSAC 匹配可视化图像
        M: 计算得到的单应性矩阵
    """
    # 使用 RANSAC 计算单应性矩阵
    # method=cv2.RANSAC: 使用 RANSAC 算法
    # ransacReprojThreshold: 重投影误差阈值
    M, mask = cv2.findHomography(
        src_pts_RT_8,
        dst_pts_RT_8,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )

    # 根据 RANSAC 结果筛选内点并可视化
    src_pts_left, src_pts_left_moved, dst_pts_right_moved, matches1, cnt = RANSAC_show(
        good_num_RT_8, src_pts_RT_8, dst_pts_RT_8, mask, M, img1, img2
    )

    # 保存 RANSAC 内点数量到全局变量
    all.cnt = cnt

    return src_pts_left, src_pts_left_moved, dst_pts_right_moved, matches1, M


# ============== RANSAC 处理和可视化 ==============

def RANSAC_show(
    good: int,
    src_pts_left: np.ndarray,
    dst_pts_right: np.ndarray,
    mask: np.ndarray,
    M: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
    根据 RANSAC 结果筛选内点并可视化

    参数:
        good: 初始匹配点数量
        src_pts_left: 源图像中的匹配点坐标
        dst_pts_right: 目标图像中的匹配点坐标
        mask: RANSAC 输出的掩码，1 表示内点，0 表示外点
        M: 计算得到的单应性矩阵
        img1: 第一张输入图像
        img2: 第二张输入图像

    返回:
        src_pts_left_: 源图像内点坐标
        src_pts_left_moved: 变换后的源图像内点坐标
        dst_pts_right_moved: 目标图像内点坐标
        matches1: RANSAC 匹配可视化图像
        cnt: 内点数量
    """
    # 将掩码展平为一维数组
    matchesMask = mask.ravel().tolist()

    # 初始化计数器和内点列表
    cnt = 0
    src_pts_left_ = []  # 源图像内点
    dst_pts_right_ = []  # 目标图像内点

    # 遍历所有匹配点，筛选内点
    for i in range(good):
        # matchesMask[i] == 1 表示该点是内点
        if matchesMask[i] == 1:
            cnt += 1  # 内点计数增加
            src_pts_left_.append(src_pts_left[i])
            dst_pts_right_.append(dst_pts_right[i])

    # 转换为 NumPy 数组
    src_pts_left_ = np.array(src_pts_left_)
    dst_pts_right_ = np.array(dst_pts_right_)

    # 初始化变换后的源点坐标数组
    src_pts_left_moved = np.zeros((cnt, 2))
    # 目标图像点不需要变换（作为参考），直接使用原始坐标
    dst_pts_right_moved = dst_pts_right_

    # 对每个内点应用单应性变换
    for i in range(cnt):
        # 计算变换后的 x 坐标
        src_pts_left_moved[i][0] = (
            M[0][0] * src_pts_left_[i][0] +
            M[0][1] * src_pts_left_[i][1] +
            M[0][2]
        )
        # 计算变换后的 y 坐标
        src_pts_left_moved[i][1] = (
            M[1][0] * src_pts_left_[i][0] +
            M[1][1] * src_pts_left_[i][1] +
            M[1][2]
        )
        # 计算分母（w 分量）
        den = (
            M[2][0] * src_pts_left_[i][0] +
            M[2][1] * src_pts_left_[i][1] +
            M[2][2]
        )
        # 归一化
        src_pts_left_moved[i][0] = src_pts_left_moved[i][0] / den
        src_pts_left_moved[i][1] = src_pts_left_moved[i][1] / den

    # 转换为 NumPy 数组
    src_pts_left_moved = np.array(src_pts_left_moved)

    # 可视化 RANSAC 匹配结果
    matches1 = draw_ransac_point(
        img1, img2, src_pts_left_moved, dst_pts_right_moved, src_pts_left_
    )

    return src_pts_left_, src_pts_left_moved, dst_pts_right_moved, matches1, cnt


def draw_ransac_point(
    src1: np.ndarray,
    src2: np.ndarray,
    src_after_pts: np.ndarray,
    dst_pts: np.ndarray,
    src_pts: np.ndarray
) -> np.ndarray:
    """
    绘制 RANSAC 匹配点可视化

    创建一张并排显示的图像，绘制匹配点和连线

    参数:
        src1: 第一张图像（源图像）
        src2: 第二张图像（目标图像）
        src_after_pts: 变换后的源图像点
        dst_pts: 目标图像点
        src_pts: 原始源图像点

    返回:
        output: 可视化图像
    """
    # 计算输出图像的高度（取两张图像的最大高度）
    height = max(src1.shape[0], src2.shape[0])

    # 计算输出图像的宽度（两张图像宽度之和）
    width = src1.shape[1] + src2.shape[1]

    # 创建输出图像（黑色背景）
    output = np.zeros((height, width, 3), dtype=np.uint8)

    # 将第一张图像放在左侧
    output[0:src1.shape[0], 0:src1.shape[1]] = src1

    # 将第二张图像放在右侧
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    # 创建颜色映射用于不同点的可视化
    # 从 0 到 255 的渐变色
    _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
    _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

    # 初始化残差总和
    residual = 0

    # 遍历所有匹配点进行绘制
    for i in range(len(src_after_pts)):
        # 计算各点坐标
        # x1, y1: 变换后的源图像点（在右侧图像上）
        x1 = int(src_after_pts[i][0] + src1.shape[1])  # 加上第一张图像宽度偏移
        y1 = int(src_after_pts[i][1])

        # x2, y2: 目标图像点（在右侧图像上）
        x2 = int(dst_pts[i][0] + src1.shape[1])
        y2 = int(dst_pts[i][1])

        # x3, y3: 原始源图像点（在左侧图像上）
        x3 = int(src_pts[i][0])
        y3 = int(src_pts[i][1])

        # 绘制变换后的源点（红色圆圈）
        cv2.circle(output, (x1, y1), 10, (0, 0, 255), -1)

        # 绘制目标点（橙色圆圈）
        cv2.circle(output, (x2, y2), 10, (0, 165, 255), -1)

        # 绘制源点（红色圆圈）
        cv2.circle(output, (x3, y3), 10, (0, 0, 255), -1)

        # 绘制连线（白色）
        cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # 累加残差（欧氏距离的平方）
        residual = (
            (src_after_pts[i][0] - dst_pts[i][0]) ** 2 +
            (src_after_pts[i][1] - dst_pts[i][1]) ** 2
        ) + residual

    # 计算平均残差并保存到全局变量
    all.Ransac_residuals = residual / len(src_after_pts)

    return output


# ============== 辅助函数 ==============

def compute_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    reprojection_threshold: float = RANSAC_REPROJ_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 计算单应性矩阵

    参数:
        src_pts: 源图像点坐标
        dst_pts: 目标图像点坐标
        reprojection_threshold: 重投影误差阈值

    返回:
        H: 单应性矩阵 (3x3)
        mask: 内点掩码 (N,)
    """
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold
    )

    return H, mask


def get_inlier_ratio(
    src_pts: np.ndarray,
    H: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 3.0
) -> float:
    """
    计算内点比例

    参数:
        src_pts: 源图像点坐标
        H: 单应性矩阵
        dst_pts: 目标图像点坐标
        threshold: 重投影误差阈值

    返回:
        内点比例 (0-1)
    """
    # 变换源点
    src_transformed = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    # 计算重投影误差
    errors = np.sqrt(
        np.sum((src_transformed - dst_pts) ** 2, axis=1)
    )

    # 统计内点数量
    inlier_count = np.sum(errors < threshold)

    # 返回内点比例
    return inlier_count / len(src_pts)
