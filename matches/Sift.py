# -*- coding: utf-8 -*-
"""
SIFT 特征匹配模块
===========
功能说明：
1. 使用 SIFT 算法检测图像特征点
2. 通过 Ratio Test 进行特征匹配
3. 返回匹配点对和匹配分数

作者：gyd
日期：2023
"""

from typing import Tuple, List  # 类型注解

import cv2  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库


# ============== SIFT 参数配置 ==============

# SIFT 特征点数量上限
SIFT_NFEATURES = 10000

# SIFT 对比度阈值（用于过滤低对比度区域）
SIFT_CONTRAST_THRESHOLD = 0.05

# Ratio Test 阈值（用于最近邻/次近邻距离比）
# 典型值在 0.7-0.8 之间
DEFAULT_RATIO_TEST = 0.75


# ============== 主要接口函数 ==============

def get_initial_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    rt: float
) -> Tuple[np.ndarray, np.ndarray, int, List[float]]:
    """
    获取两张图像的初始 SIFT 匹配

    这是模块的主要入口函数，执行完整的 SIFT 特征检测和匹配流程

    参数:
        img1: 第一张输入图像（参考图像）
        img2: 第二张输入图像（候选图像）
        rt: Ratio Test 阈值，用于过滤低质量匹配
            - 较小的值（如 0.7）会保留更严格的匹配
            - 较大的值（如 0.85）会保留更多的匹配

    返回:
        src_pts: 源图像中的匹配点坐标，形状为 (N, 2)
        dst_pts: 目标图像中的匹配点坐标，形状为 (N, 2)
        good_num: 匹配点的数量
        scores: 匹配分数列表（最近邻/次近邻距离比）

    示例:
        >>> src, dst, num, scores = get_initial_matches(img1, img2, 0.75)
        >>> print(f"找到 {num} 个匹配点")
    """
    # 执行 SIFT 特征检测
    matcher, kp1, kp2, desc1, desc2 = siftTest(img1, img2)

    # 执行特征匹配
    src_pts, dst_pts, good_num, scores = get_pts(
        matcher, kp1, kp2, desc1, desc2, rt
    )

    return src_pts, dst_pts, good_num, scores


# ============== SIFT 特征检测 ==============

def siftTest(
    img1: np.ndarray,
    img2: np.ndarray
) -> Tuple[cv2.BFMatcher, List, List, np.ndarray, np.ndarray]:
    """
    在两张图像上执行 SIFT 特征检测

    参数:
        img1: 第一张输入图像
        img2: 第二张输入图像

    返回:
        matcher: BFMatcher 对象，用于后续匹配
        kp1: 第一张图像的关键点列表
        kp2: 第二张图像的关键点列表
        desc1: 第一张图像的描述符
        desc2: 第二张图像的描述符
    """
    # 创建 SIFT 特征检测器
    # nfeatures: 最大特征点数量
    # contrastThreshold: 对比度阈值，用于过滤低对比度区域
    # sift论文用的 0.03，nOctaveLayers 若为 3，设置参数为 0.09
    # 这里使用 10000 个特征点和 0.05 的对比度阈值
    sift = cv2.SIFT_create(
        nfeatures=SIFT_NFEATURES,
        contrastThreshold=SIFT_CONTRAST_THRESHOLD
    )

    # 在第一张图像中检测关键点和计算描述符
    # None 表示不使用 mask 检测整张图像
    kp1, desc1 = sift.detectAndCompute(img1, None)

    # 在第二张图像中检测关键点和计算描述符
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # 创建 BFMatcher（暴力匹配器）
    # 使用 NORM_L2 距离，这是 SIFT 描述符的标准距离度量
    bf = cv2.BFMatcher()

    return bf, kp1, kp2, desc1, desc2


# ============== 特征匹配 ==============

def get_pts(
    matcher: cv2.BFMatcher,
    kp1: List,
    kp2: List,
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float
) -> Tuple[np.ndarray, np.ndarray, int, List[float]]:
    """
    执行特征匹配并返回匹配点对

    使用 KNN (K-Nearest Neighbors) 匹配，然后应用 Ratio Test 过滤

    参数:
        matcher: BFMatcher 对象
        kp1: 第一张图像的关键点列表
        kp2: 第二张图像的关键点列表
        desc1: 第一张图像的描述符
        desc2: 第二张图像的描述符
        ratio_threshold: Ratio Test 阈值

    返回:
        src_pts_left: 源图像匹配点坐标，形状为 (N, 2)
        dst_pts_right: 目标图像匹配点坐标，形状为 (N, 2)
        good_num: 匹配点的数量
        scores: 匹配分数列表
    """
    # 执行 KNN 匹配，k=2 表示为每个描述符找 2 个最近邻
    # 返回的 raw_matches 是一个列表，每个元素包含两个匹配结果
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    # 用于存储通过的匹配和分数
    good = []  # 通过 Ratio Test 的匹配
    scores = []  # 匹配分数（最近邻/次近邻距离比）

    # 遍历所有匹配结果
    for m, n in raw_matches:
        # Ratio Test: 如果最近邻距离小于阈值 * 次近邻距离
        # 则认为这是一个好的匹配
        if m.distance < ratio_threshold * n.distance:
            # 计算并保存匹配分数
            scores.append(m.distance / n.distance)
            # 保存好的匹配
            good.append(m)

    # 提取匹配点的坐标
    # 从关键点中提取 (x, y) 坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    # 为后续处理准备点坐标列表
    # 这些列表会用于一些需要列表格式的函数
    npts1 = []  # 源图像点坐标列表
    npts2 = []  # 目标图像点坐标列表

    # 遍历第一张图像的所有关键点
    for keypoint in kp1:
        npts1.append((keypoint.pt[0], keypoint.pt[1]))

    # 遍历第二张图像的所有关键点
    for keypoint in kp2:
        npts2.append((keypoint.pt[0], keypoint.pt[1]))

    # 从关键点中提取匹配的坐标（使用 queryIdx 和 trainIdx）
    src_pts_left = np.float32([npts1[m.queryIdx] for m in good])  # 源图像点
    dst_pts_right = np.float32([npts2[m.trainIdx] for m in good])  # 目标图像点

    return src_pts_left, dst_pts_right, len(good), scores


# ============== 辅助函数 ==============

def compute_match_statistics(scores: List[float]) -> dict:
    """
    计算匹配分数的统计信息

    参数:
        scores: 匹配分数列表

    返回:
        包含统计信息的字典
    """
    if not scores:
        return {
            'count': 0,
            'mean': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0,
        }

    import statistics

    return {
        'count': len(scores),
        'mean': statistics.mean(scores),
        'min': min(scores),
        'max': max(scores),
        'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }


def filter_matches_by_distance(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    scores: List[float],
    max_distance: float
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    根据距离过滤匹配点

    参数:
        src_pts: 源图像匹配点坐标
        dst_pts: 目标图像匹配点坐标
        scores: 匹配分数列表
        max_distance: 最大允许的距离比

    返回:
        过滤后的 src_pts, dst_pts, scores
    """
    # 将分数转换为 NumPy 数组
    scores_array = np.array(scores)

    # 找到满足阈值条件的索引
    valid_indices = scores_array <= max_distance

    # 过滤匹配点
    filtered_src = src_pts[valid_indices]
    filtered_dst = dst_pts[valid_indices]
    filtered_scores = scores_array[valid_indices].tolist()

    return filtered_src, filtered_dst, filtered_scores
