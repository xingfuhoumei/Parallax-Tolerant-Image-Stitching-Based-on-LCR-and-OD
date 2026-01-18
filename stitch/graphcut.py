# -*- coding: utf-8 -*-
"""
图分割与图像融合模块
===========
功能说明：
1. 基于图割算法的最优接缝查找
2. 多频带拉普拉斯金字塔融合
3. 色差校正与增益补偿
4. 图像拼接后处理优化

主要算法：
- MaxFlow 图分割算法
- 拉普拉斯金字塔多频带融合
- LAB 色彩空间增益补偿
- 高斯权重与边缘保持平滑

作者：gyd
日期：2023
"""

from typing import Tuple, Optional, List  # 类型注解

import cv2 as cv  # OpenCV 图像处理库
import maxflow  # 最大流/最小割算法库
import numpy as np  # NumPy 数值计算库
from numba import jit  # JIT 编译加速
from PIL import Image  # PIL 图像处理库


# ============== 配置常量 ==============

# 图分割参数
SOURCE_CAPACITY = (np.inf, 0)  # 源点容量（无限容量）
SINK_CAPACITY = (0, np.inf)    # 汇点容量（无限容量）

# 多频带融合参数
DEFAULT_PYRAMID_LEVELS = 5     # 默认金字塔层数
GAUSSIAN_KERNEL_SIZE = 5       # 高斯核大小

# 色彩校正参数
GAIN_CLIP_RANGE = (0.5, 2.0)   # 增益裁剪范围
OFFSET_CLIP_L = 30             # L 通道偏移裁剪
OFFSET_CLIP_AB = 20            # A/B 通道偏移裁剪


# ============== 主要功能函数：图分割拼接 ==============

def graph_cut(
    result_img1: np.ndarray,
    result_img2: np.ndarray,
    weight: np.ndarray,
    weight1: np.ndarray,
    a: int,
    b: int,
    m: int,
    n: int,
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用图割算法进行图像拼接

    该函数实现基于最大流/最小割算法的图像拼接：
    1. 构建图网络，每个像素是一个节点
    2. 设置源点和汇点
    3. 计算最小割，得到最优接缝
    4. 根据分割结果融合两张图像

    参数:
        result_img1: 第一张对齐后的图像
        result_img2: 第二张对齐后的图像
        weight: 能量权重矩阵（浮点型）
        weight1: 能量权重矩阵（uint8型）
        a: 重叠区域起始行
        b: 重叠区域结束列（用于判断拼接方向）
        m: 拼接图像起始列
        n: 拼接图像结束列
        img1: 原始第一张图像
        img2: 原始第二张图像
        H: 单应性矩阵

    返回:
        overlap: 融合后的重叠区域
        overlap1: 带接缝可视化图
        seam_output: 接缝输出图
        H4stitching: 最终拼接结果
    """
    # ========== 步骤1：转换为灰度图 ==========
    # 图割算法基于单通道灰度图计算
    result_img1_grey = cv.cvtColor(result_img1, cv.COLOR_BGR2GRAY)
    result_img2_grey = cv.cvtColor(result_img2, cv.COLOR_BGR2GRAY)

    # ========== 步骤2：创建最大流图 ==========
    # 使用 float 类型创建图，支持小数权重
    g = maxflow.Graph[float]()

    # 添加网格节点，每个像素对应一个节点
    nodes = g.add_grid_nodes(weight.shape)

    # ========== 步骤3：根据拼接方向处理 ==========
    # b == 0 表示候选图像在右侧（从左到右拼接）
    if b == 0:
        # --- 情况1：候选图像在右侧 ---
        # 查找上下白边（有效区域边界）
        # fixme: 这个函数的具体含义需要进一步确认
        i, j = get_white_left(weight_find=weight)

        # 设置源点和汇点
        source, sink = SOURCE_CAPACITY, SINK_CAPACITY

        # 添加边界边：右列连接到汇点（顶部右侧）
        g.add_grid_tedges(nodes[i:j + 1, -1], *sink)

        # 添加边界边：除最后一行外，左列连接到源点（底部左侧）
        g.add_grid_tedges(nodes[:-1, 0], *source)

    else:
        # --- 情况2：候选图像在左侧 ---
        # 查找上下白边
        i, j = get_white_right(weight_find=weight)

        # 设置源点和汇点
        source, sink = SOURCE_CAPACITY, SINK_CAPACITY

        # 添加边界边：所有行，右列连接到汇点（顶部右侧）
        g.add_grid_tedges(nodes[0:, -1], *sink)

        # 添加边界边：左列指定行范围连接到源点（底部左侧）
        g.add_grid_tedges(nodes[i:j + 1, 0], *source)

    # ========== 步骤4：添加邻域边 ==========
    # 定义邻域结构：只连接右方和下方邻居
    # 结构矩阵中 1 表示存在连接
    structure = np.array([
        [0, 0, 0],
        [0, 0, 1],  # 连接右方邻居
        [0, 1, 0]   # 连接下方邻居
    ])

    # 添加水平方向的边（左右连接）
    # np.roll 将数组向左移动一位，实现相邻列的连接
    g.add_grid_edges(
        nodes,
        weights=(weight + np.roll(weight, -1, axis=1)),  # 权重为相邻像素权重之和
        structure=structure,
        symmetric=True  # 对称添加
    )

    # 添加垂直方向的边（上下连接）
    g.add_grid_edges(
        nodes,
        weights=(weight + np.roll(weight, -1, axis=0)),  # 权重为相邻像素权重之和
        structure=structure,
        symmetric=True
    )

    # ========== 步骤5：计算最大流/最小割 ==========
    max_flow = g.maxflow()

    # ========== 步骤6：获取分割结果 ==========
    # get_grid_segments 返回布尔数组，True 表示属于源点集，False 表示属于汇点集
    segments = g.get_grid_segments(nodes)

    # ========== 步骤7：生成拼接画布 ==========
    H4stitching = get_stitched_image(img2, img1, H)

    # ========== 步骤8：创建重叠区域输出 ==========
    # 初始化重叠区域为单通道，然后扩展为三通道
    overlap = np.zeros_like(segments, dtype=np.uint8)
    overlap = np.stack([overlap] * 3, axis=2)  # 扩展为 BGR 三通道

    # 初始化接缝输出
    seam_output = np.zeros(overlap.shape)

    # ========== 步骤9：检测接缝边界 ==========
    # 使用逻辑异或操作检测接缝（分割结果变化的地方）
    # np.roll 用于比较相邻像素
    seam = np.logical_or.reduce([
        np.logical_xor(segments, np.roll(segments, 1, axis=0)),   # 上下相邻
        np.logical_xor(segments, np.roll(segments, 1, axis=1)),   # 左右相邻
        np.logical_xor(segments, np.roll(segments, -1, axis=0)),  # 上下相邻（反向）
        np.logical_xor(segments, np.roll(segments, -1, axis=1))   # 左右相邻（反向）
    ])

    # 清除图像边界的接缝标记
    seam[:, :1] = seam[:1, :] = seam[:, -1:] = seam[-1:, :] = False

    # ========== 步骤10：根据分割结果融合图像 ==========
    if b == 0:
        # 候选图像在右侧：img1 在左，img2 在右
        overlap, seam_output = combine_overlap1(
            segments, result_img1, result_img2, overlap, seam, seam_output
        )
    else:
        # 候选图像在左侧：img2 在左，img1 在右
        overlap, seam_output = combine_overlap1(
            segments, result_img2, result_img1, overlap, seam, seam_output
        )

    # ========== 步骤11：生成带接缝的可视化图 ==========
    # 在接缝位置标记灰色（150）
    overlap1 = np.where(seam, [[150]], weight1)
    overlap1 = np.array(overlap1, dtype='uint8')

    # ========== 步骤12：将融合结果放入最终图像 ==========
    # 矩阵操作后一定要转换为 uint8
    seam_output = np.array(seam_output, dtype='uint8')

    # 获取重叠区域尺寸
    w1, h1 = overlap.shape[:2]

    # 根据拼接方向放置融合结果
    if b == 0:
        # 从 (a, m) 开始放置
        H4stitching[a:w1 + a, m:n + 1] = overlap
    else:
        # 从 (a, b) 开始放置
        H4stitching[a:w1 + a, b:h1 + b] = overlap

    return overlap, overlap1, seam_output, H4stitching


def get_stitched_image(
    img1: np.ndarray,
    img2: np.ndarray,
    M: np.ndarray
) -> np.ndarray:
    """
    生成完整的拼接图像

    将两张图像通过单应性矩阵拼接在一起

    参数:
        img1: 第一张图像（参考图像）
        img2: 第二张图像（待变换图像）
        M: 3x3 单应性矩阵

    返回:
        result_img: 拼接后的完整图像
    """
    # 获取输入图像的高度和宽度
    w1, h1 = img1.shape[:2]  # img1 的高度、宽度
    w2, h2 = img2.shape[:2]  # img2 的高度、宽度

    # 定义 img1 的四个角点坐标
    # 顺序：左上、右上、右下、左下
    img1_dims = np.float32([
        [0, 0],      # 左上角
        [0, w1],     # 右上角
        [h1, w1],    # 右下角
        [h1, 0]      # 左下角
    ]).reshape(-1, 1, 2)

    # 定义 img2 的四个角点坐标
    img2_dims_temp = np.float32([
        [0, 0],      # 左上角
        [0, w2],     # 右上角
        [h2, w2],    # 右下角
        [h2, 0]      # 左下角
    ]).reshape(-1, 1, 2)

    # 对 img2 的角点应用透视变换
    # 得到变换后的坐标
    img2_dims = cv.perspectiveTransform(img2_dims_temp, M)

    # 合并所有角点，计算输出画布的大小
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # 计算画布边界
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # 计算平移量（将画布移到正坐标区域）
    transform_dist = [-x_min, -y_min]

    # 构建平移矩阵
    transform_array = np.array([
        [1, 0, transform_dist[0]],  # X 方向平移
        [0, 1, transform_dist[1]],  # Y 方向平移
        [0, 0, 1]                   # 齐次坐标
    ])

    # 执行透视变换：先单应性，再平移
    result_img = cv.warpPerspective(
        img2,
        transform_array.dot(M),
        (x_max - x_min, y_max - y_min)
    )

    # 将 img1 放入结果图像的对应位置
    result_img[
        transform_dist[1]:w1 + transform_dist[1],  # 行范围
        transform_dist[0]:h1 + transform_dist[0]   # 列范围
    ] = img1

    return np.array(result_img)


@jit(nopython=True)
def get_white_left(weight_find: np.ndarray) -> Tuple[int, int]:
    """
    从左侧查找白色（高权重）区域边界（使用 JIT 加速）

    用于确定图割算法中源点和汇点的连接范围

    参数:
        weight_find: 权重矩阵，uint8 类型

    返回:
        i: 上边界行索引
        j: 下边界行索引

    检测逻辑：
    - 从上往下扫描倒数第二列，查找第一个低权重像素
    - 从下往上扫描倒数第二列，查找第一个低权重像素
    """
    # 获取矩阵的行数和列数
    col = weight_find.shape[0]  # 行数
    row = weight_find.shape[1]  # 列数

    # 从上往下扫描，查找上边界
    for i in range(col):
        # 检查倒数第二列的权重值
        if weight_find[i, row - 2] > 140:
            # 权重大于 140，继续向下扫描
            continue
        else:
            # 权重小于等于 140，找到上边界
            break

    # 从下往上扫描，查找下边界
    for j in range(col, -1, -1):
        # 检查倒数第二列的权重值
        if weight_find[j - 1, row - 2] > 140:
            # 权重大于 140，继续向上扫描
            continue
        else:
            # 权重小于等于 140，找到下边界
            break

    return i, j


@jit(nopython=True)
def get_white_right(weight_find: np.ndarray) -> Tuple[int, int]:
    """
    从右侧查找白色（高权重）区域边界（使用 JIT 加速）

    用于确定图割算法中源点和汇点的连接范围

    参数:
        weight_find: 权重矩阵，uint8 类型

    返回:
        i: 上边界行索引
        j: 下边界行索引

    检测逻辑：
    - 从上往下扫描第二列，查找第一个低权重像素
    - 从下往上扫描第二列，查找第一个低权重像素
    """
    # 获取矩阵的行数和列数
    col = weight_find.shape[0]  # 行数
    row = weight_find.shape[1]  # 列数

    # 从上往下扫描，查找上边界
    for i in range(col):
        # 检查第二列的权重值
        if weight_find[i, 1] > 140:
            # 权重大于 140，继续向下扫描
            continue
        else:
            # 权重小于等于 140，找到上边界
            break

    # 从下往上扫描，查找下边界
    for j in range(col, -1, -1):
        # 检查第二列的权重值
        if weight_find[j - 1, 1] > 140:
            # 权重大于 140，继续向上扫描
            continue
        else:
            # 权重小于等于 140，找到下边界
            break

    return i, j


@jit(nopython=True)
def combine_overlap1(
    segments: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    overlap: np.ndarray,
    seam: np.ndarray,
    seam_output: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据图割分割结果融合两张图像（使用 JIT 加速）

    参数:
        segments: 图割分割结果，布尔数组
                  True 表示使用 img1，False 表示使用 img2
        img1: 第一张图像
        img2: 第二张图像
        overlap: 输出的融合图像
        seam: 接缝位置，布尔数组
        seam_output: 带接缝标记的输出图像

    返回:
        overlap: 融合后的图像
        seam_output: 带绿色接缝标记的图像
    """
    # 获取图像的行数和列数
    rows = segments.shape[0]
    cols = segments.shape[1]

    # ========== 步骤1：根据分割结果融合图像 ==========
    for i in range(rows):
        for j in range(cols):
            if not segments[i][j]:
                # 分割结果为 False，使用 img2 的像素
                overlap[i][j] = img2[i][j]
            else:
                # 分割结果为 True，使用 img1 的像素
                overlap[i][j] = img1[i][j]

    # ========== 步骤2：生成带接缝的可视化图 ==========
    for i in range(seam.shape[0]):
        for j in range(seam.shape[1]):
            if not seam[i][j]:
                # 非接缝区域，使用融合后的像素
                seam_output[i][j] = overlap[i][j]
            else:
                # 接缝区域，标记为绿色 [0, 255, 0]
                seam_output[i][j] = [0, 255, 0]

    return overlap, seam_output


# ============== 色彩校正与增益补偿模块 ==============

def gain_compensation(
    img_source: np.ndarray,
    img_target: np.ndarray,
    overlap_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    增益补偿 (Gain Compensation) 算法

    基于 Brown & Lowe 2007 论文的增强版本
    这是解决色差问题的 foundational step

    参数:
        img_source: 待校正的源图像
        img_target: 目标参考图像
        overlap_mask: 重叠区域掩码（可选）

    返回:
        img_corrected: 色彩校正后的图像
    """
    # 如果没有提供重叠区域掩码，使用整个图像
    if overlap_mask is None:
        h, w = img_source.shape[:2]
        overlap_mask = np.ones((h, w), dtype=np.uint8) * 255

    # 检查有效区域
    valid_mask = overlap_mask > 0
    if not np.any(valid_mask):
        return img_source

    # ========== 步骤1：在 LAB 色彩空间进行初步校正 ==========
    # 转换到 LAB 空间（更接近人类感知）
    img_source_lab = cv.cvtColor(img_source, cv.COLOR_BGR2LAB).astype(np.float64)
    img_target_lab = cv.cvtColor(img_target, cv.COLOR_BGR2LAB).astype(np.float64)

    # 提取重叠区域
    source_overlap = img_source_lab[valid_mask]
    target_overlap = img_target_lab[valid_mask]

    result_lab = img_source_lab.copy()

    # ========== 步骤2：对 LAB 三个通道分别处理 ==========
    for channel in range(3):
        source_channel = source_overlap[:, channel]
        target_channel = target_overlap[:, channel]

        # 筛选有效像素
        if channel == 0:  # L 通道（亮度）
            valid_pixels = (
                (source_channel > 10) & (source_channel < 90) &
                (target_channel > 10) & (target_channel < 90)
            )
        else:  # A 和 B 通道（色彩）
            valid_pixels = (
                (source_channel > 110) & (source_channel < 150) &
                (target_channel > 110) & (target_channel < 150)
            )

        # 如果有效像素太少，使用所有像素
        if np.sum(valid_pixels) < 50:
            valid_pixels = np.ones(len(source_channel), dtype=bool)

        source_valid = source_channel[valid_pixels]
        target_valid = target_channel[valid_pixels]

        if len(source_valid) == 0:
            continue

        # ========== 步骤3：使用多种统计方法计算校正参数 ==========
        # 方法1：均值匹配
        source_mean = np.mean(source_valid)
        target_mean = np.mean(target_valid)
        mean_diff = target_mean - source_mean

        # 方法2：中位数匹配（更鲁棒）
        source_median = np.median(source_valid)
        target_median = np.median(target_valid)
        median_diff = target_median - source_median

        # 方法3：分位数匹配
        source_q25 = np.percentile(source_valid, 25)
        source_q75 = np.percentile(source_valid, 75)
        target_q25 = np.percentile(target_valid, 25)
        target_q75 = np.percentile(target_valid, 75)

        # 计算增益和偏移
        source_range = source_q75 - source_q25
        target_range = target_q75 - target_q25

        if source_range > 1e-6:
            gain = target_range / source_range
            gain = np.clip(gain, GAIN_CLIP_RANGE[0], GAIN_CLIP_RANGE[1])
        else:
            gain = 1.0

        # 综合多种方法计算偏移（加权平均）
        offset_mean = mean_diff
        offset_median = median_diff
        offset_percentile = target_q25 - gain * source_q25

        # 中位数权重更高（更鲁棒）
        offset = 0.2 * offset_mean + 0.6 * offset_median + 0.2 * offset_percentile

        # 限制偏移范围
        if channel == 0:  # L 通道
            offset = np.clip(offset, -OFFSET_CLIP_L, OFFSET_CLIP_L)
        else:  # A 和 B 通道
            offset = np.clip(offset, -OFFSET_CLIP_AB, OFFSET_CLIP_AB)

        # ========== 步骤4：应用变换 ==========
        result_lab[:, :, channel] = gain * img_source_lab[:, :, channel] + offset

    # ========== 步骤5：确保 LAB 值在有效范围内 ==========
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)   # L 通道
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -127, 127)  # A 通道
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -127, 127)  # B 通道

    # ========== 步骤6：转换回 BGR ==========
    result_bgr = cv.cvtColor(result_lab.astype(np.uint8), cv.COLOR_LAB2BGR)

    # ========== 步骤7：在 BGR 空间进行微调 ==========
    result_bgr_float = result_bgr.astype(np.float64)
    img_target_float = img_target.astype(np.float64)

    # 计算整体色温调整
    result_mean = np.mean(result_bgr_float[valid_mask], axis=0)
    target_mean = np.mean(img_target_float[valid_mask], axis=0)

    # 轻微调整以进一步匹配（30% 的调整强度）
    color_diff = target_mean - result_mean
    adjustment = color_diff * 0.3

    result_bgr_float += adjustment

    return np.clip(result_bgr_float, 0, 255).astype(np.uint8)


# ============== 多频带拉普拉斯金字塔融合模块 ==============

def build_gaussian_pyramid(
    img: np.ndarray,
    levels: int
) -> List[np.ndarray]:
    """
    构建高斯金字塔

    参数:
        img: 输入图像
        levels: 金字塔层数

    返回:
        pyramid: 高斯金字塔列表，从底层到顶层
    """
    pyramid = [img]
    current = img.copy()

    for _ in range(levels - 1):
        # 下采样：先模糊再缩小
        current = cv.pyrDown(current)
        pyramid.append(current)

    return pyramid


def build_laplacian_pyramid(
    gaussian_pyramid: List[np.ndarray]
) -> List[np.ndarray]:
    """
    从高斯金字塔构建拉普拉斯金字塔

    参数:
        gaussian_pyramid: 高斯金字塔列表

    返回:
        laplacian_pyramid: 拉普拉斯金字塔列表
    """
    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):
        # 上采样下一层
        expanded = cv.pyrUp(gaussian_pyramid[i + 1])

        # 确保尺寸匹配
        if expanded.shape[:2] != gaussian_pyramid[i].shape[:2]:
            expanded = cv.resize(
                expanded,
                (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            )

        # 计算拉普拉斯层（当前层减去上采样的上层）
        laplacian = gaussian_pyramid[i] - expanded
        laplacian_pyramid.append(laplacian)

    # 添加最后一层（最小的层）
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def reconstruct_from_laplacian_pyramid(
    laplacian_pyramid: List[np.ndarray]
) -> np.ndarray:
    """
    从拉普拉斯金字塔重建图像

    参数:
        laplacian_pyramid: 拉普拉斯金字塔列表

    返回:
        reconstructed: 重建的图像
    """
    current = laplacian_pyramid[-1].copy()

    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # 上采样当前层
        current = cv.pyrUp(current)

        # 确保尺寸匹配
        if current.shape[:2] != laplacian_pyramid[i].shape[:2]:
            current = cv.resize(
                current,
                (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            )

        # 添加拉普拉斯层
        current = current + laplacian_pyramid[i]

    return current


def laplacian_pyramid_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 4
) -> np.ndarray:
    """
    拉普拉斯金字塔融合

    产生无缝的拼接结果

    参数:
        img1: 第一张图像
        img2: 第二张图像
        levels: 金字塔层数

    返回:
        blended: 融合后的图像
    """
    h, w = img1.shape[:2]

    # 创建混合 mask（中心权重更高）
    mask = np.ones((h, w), dtype=np.float32)
    center_w = w // 2

    # 创建从左到右的渐变权重
    for x in range(w):
        if x < center_w:
            weight = 0.3 + 0.4 * (x / center_w)  # 0.3 -> 0.7
        else:
            weight = 0.7 + 0.3 * ((w - 1 - x) / (w - center_w))  # 0.7 -> 1.0
        mask[:, x] = weight

    # 构建高斯金字塔
    pyr1 = build_gaussian_pyramid(img1.astype(np.float32), levels)
    pyr2 = build_gaussian_pyramid(img2.astype(np.float32), levels)
    mask_pyr = build_gaussian_pyramid(mask, levels)

    # 构建拉普拉斯金字塔
    lap1 = build_laplacian_pyramid(pyr1)
    lap2 = build_laplacian_pyramid(pyr2)

    # 混合拉普拉斯金字塔
    blended_lap = []
    for i in range(levels):
        if i < len(mask_pyr):
            m = mask_pyr[i]
            if len(m.shape) == 2:
                m = np.stack([m] * 3, axis=2)
            blended_lap.append(lap1[i] * m + lap2[i] * (1 - m))
        else:
            blended_lap.append(lap1[i] * 0.5 + lap2[i] * 0.5)

    # 重建图像
    result = reconstruct_from_laplacian_pyramid(blended_lap)

    return np.clip(result, 0, 255).astype(np.uint8)


# ============== 辅助功能函数 ==============

def white_balance_correction(
    source: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    简单的白平衡校正

    参数:
        source: 待校正图像
        reference: 参考图像

    返回:
        corrected: 白平衡校正后的图像
    """
    source_float = source.astype(np.float32)
    reference_float = reference.astype(np.float32)

    # 计算每个通道的平均值
    source_mean = np.mean(source_float, axis=(0, 1))
    reference_mean = np.mean(reference_float, axis=(0, 1))

    # 计算校正比例，避免除零
    ratios = np.where(source_mean > 1, reference_mean / source_mean, 1.0)

    # 应用校正
    corrected = source_float * ratios

    return np.clip(corrected, 0, 255).astype(np.uint8)


def gradient_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    blend_width: int = 50
) -> np.ndarray:
    """
    创建渐变融合效果以减少接缝处的色差

    参数:
        img1: 第一张图像
        img2: 第二张图像
        blend_width: 融合带宽度

    返回:
        blended: 渐变融合后的图像
    """
    h, w = img1.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    # 创建从左到右的渐变权重
    if w > blend_width * 2:
        # 左边缘渐变
        for i in range(min(blend_width, w // 2)):
            mask[:, i] = i / blend_width
        # 右边缘渐变
        for i in range(max(0, w - blend_width), w):
            mask[:, i] = (w - 1 - i) / blend_width

    # 扩展 mask 到 3 个通道
    mask = np.stack([mask] * 3, axis=2)

    # 进行加权混合
    blended = (img1.astype(np.float32) * mask +
               img2.astype(np.float32) * (1 - mask))

    return np.uint8(np.clip(blended, 0, 255))


def color_match(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    使用直方图匹配进行色彩校正

    参数:
        source: 待校正图像
        target: 目标图像

    返回:
        matched: 色彩匹配后的图像
    """
    # 转换到 LAB 空间
    source_lab = cv.cvtColor(source, cv.COLOR_BGR2LAB)
    target_lab = cv.cvtColor(target, cv.COLOR_BGR2LAB)

    matched_lab = np.zeros_like(source_lab)

    # 对每个 LAB 通道进行直方图匹配
    for i in range(3):
        # 计算直方图
        source_hist = cv.calcHist([source_lab], [i], None, [256], [0, 256])
        target_hist = cv.calcHist([target_lab], [i], None, [256], [0, 256])

        # 计算累积分布函数
        source_cdf = source_hist.cumsum()
        target_cdf = target_hist.cumsum()

        # 标准化
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]

        # 创建查找表
        lut = np.zeros(256, dtype=np.uint8)
        for j in range(256):
            idx = np.argmin(np.abs(target_cdf - source_cdf[j]))
            lut[j] = idx

        # 应用查找表
        matched_lab[:, :, i] = cv.LUT(source_lab[:, :, i], lut)

    # 转换回 BGR
    return cv.cvtColor(matched_lab, cv.COLOR_LAB2BGR)


# ============== 模块入口（用于测试）=============

if __name__ == "__main__":
    # 模块测试代码
    print("图分割与图像融合模块")
    print("使用方法：from stitch.graphcut import graph_cut, gain_compensation")
