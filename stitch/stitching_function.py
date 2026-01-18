# -*- coding: utf-8 -*-
"""
图像拼接几何变换模块
===========
功能说明：
1. 透视变换与图像对齐
2. 自动计算输出画布尺寸
3. 重叠区域精确定位
4. 图像裁剪与结果提取

主要算法：
- 单应性矩阵透视变换
- 边界框计算与画布扩展
- 重叠区域智能检测

作者：gyd
日期：2023
"""

from typing import Tuple, List  # 类型注解
import warnings  # 警告处理

import cv2 as cv  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库
from numba import jit  # JIT 编译加速


# ============== 配置常量 ==============

# 透视变换时的边界扩展比例
CANVAS_PADDING = 0.5  # 扩展 0.5 像素的边界

# 重叠区域检测的最小像素阈值
MIN_OVERLAP_THRESHOLD = 10  # 最小重叠像素数


# ============== 主要功能函数 ==============

def get_warp(
    img1: np.ndarray,
    img2: np.ndarray,
    M: np.ndarray
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """
    对候选图像进行透视变换并裁剪对齐

    该函数执行以下操作：
    1. 计算变换后图像的画布大小
    2. 应用透视变换到 img2
    3. 裁剪出与 img1 对齐的区域

    参数:
        img1: 候选图像（待变换的图像）
        img2: 参考图像（基准图像）
        M: 3x3 单应性矩阵，用于透视变换

    返回:
        result_img2: 裁剪后的变换图像，与 img1 尺寸相同
        offset_y: Y 方向的偏移量（画布中的起始行）
        offset_x: X 方向的偏移量（画布中的起始列）
        full_warped: 完整的透视变换图像（未裁剪）

    示例:
        >>> warped_img, y_off, x_off, full_img = get_warp(img1, img2, H)
    """
    # 获取输入图像的高度和宽度
    # 注意：OpenCV 中 shape 返回 (行数, 列数, 通道数)
    # 行数对应高度，列数对应宽度
    w1, h1 = img1.shape[:2]  # 候选图像的高度和宽度
    w2, h2 = img2.shape[:2]  # 参考图像的高度和宽度

    # 定义候选图像的四个角点坐标
    # 顺序：左上、右上、右下、左下
    # 格式：[行, 列] 即 [y, x]
    img1_dims = np.float32([
        [0, 0],      # 左上角 (行=0, 列=0)
        [0, w1],     # 右上角 (行=0, 列=宽度)
        [w1, w1],    # 右下角 - 原代码可能有误，应为 [h1, w1]
        [h1, 0]      # 左下角 (行=高度, 列=0)
    ]).reshape(-1, 1, 2)  # 重塑为 (4, 1, 2) 形状，每个点是一个 1x2 矩阵

    # 定义参考图像的四个角点坐标（与 img1 相同的顺序）
    img2_dims_temp = np.float32([
        [0, 0],      # 左上角
        [0, w2],     # 右上角
        [w2, w2],    # 右下角
        [h2, 0]      # 左下角
    ]).reshape(-1, 1, 2)

    # 对参考图像的角点应用单应性变换
    # 得到在候选图像坐标系中的对应位置
    img2_dims = cv.perspectiveTransform(img2_dims_temp, M)

    # 合并两个图像的角点坐标，计算整体边界
    # concatenate 沿着第 0 维（样本维度）连接
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # 计算变换后画布的最小和最大坐标
    # min(axis=0) 找到所有点中行、列的最小值
    # ravel() 将二维数组展平为一维
    # int32() 转换为整数坐标
    # -0.5 和 +0.5 是为了四舍五入到最近的整数
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # 计算平移变换矩阵
    # 用于将变换后的图像平移到画布的正方向区域
    # -x_min, -y_min 将最小坐标平移到 (0, 0)
    transform_dist = [-x_min, -y_min]

    # 构建 3x3 平移矩阵
    # 格式：[[1, 0, dx], [0, 1, dy], [0, 0, 1]]
    # dx 和 dy 是平移量
    transform_array = np.array([
        [1, 0, transform_dist[0]],  # X 方向平移
        [0, 1, transform_dist[1]],  # Y 方向平移
        [0, 0, 1]                   # 齐次坐标（保持为 1）
    ])

    # 执行透视变换：先应用单应性矩阵 M，再平移
    # dot() 进行矩阵乘法，组合两个变换
    # warpPerspective 参数：
    # - src: 源图像
    # - M: 变换矩阵
    # - dsize: 输出图像尺寸 (宽度, 高度)
    result_img = cv.warpPerspective(
        img2,                      # 源图像（参考图像）
        transform_array.dot(M),    # 组合变换矩阵（平移 + 单应性）
        (x_max - x_min, y_max - y_min)  # 输出画布尺寸
    )

    # 从变换后的图像中提取与候选图像对齐的区域
    # 这个区域与原始 img1 具有相同的尺寸
    result_img2 = result_img[
        transform_dist[1]:w1 + transform_dist[1],  # 行范围：Y 偏移到 Y 偏移 + 高度
        transform_dist[0]:h1 + transform_dist[0]   # 列范围：X 偏移到 X 偏移 + 宽度
    ]

    return result_img2, transform_dist[1], transform_dist[0], result_img


def find_overlap_region(
    result_img1: np.ndarray,
    result_img2: np.ndarray,
    overlap_mask: np.ndarray,
    has_mask: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """
    在两张对齐的图像中找到重叠区域

    通过检测图像中的非零像素区域，确定两张图像的有效重叠范围

    参数:
        result_img1: 第一张对齐后的图像
        result_img2: 第二张对齐后的图像
        overlap_mask: 视差掩码或辅助掩码（可选）
        has_mask: 是否使用掩码进行重叠区域检测

    返回:
        resize_img1: 裁剪到重叠区域的第一张图像
        resize_img2: 裁剪到重叠区域的第二张图像
        col_start: 重叠区域的起始列索引
        col_end: 重叠区域的结束列索引
        overlap_mask: （可选）裁剪后的掩码
    """
    # 检测第一张图像的重叠列范围
    # get_w 返回 (起始列, 结束列)
    a, b = get_w(result_img1)

    # 根据检测结果裁剪图像到重叠区域
    # 使用列切片：从 a 列到 b+1 列（Python 切片右边界不包含，所以 +1）
    resize_img1 = result_img1[:, a:b + 1]  # 裁剪第一张图像
    resize_img2 = result_img2[:, a:b + 1]  # 裁剪第二张图像

    # 如果提供了掩码且需要使用
    if has_mask:
        # 同样裁剪掩码到重叠区域
        overlap_mask = overlap_mask[:, a:b + 1]

    return resize_img1, resize_img2, a, b, overlap_mask


def get_canvas_size(
    img1: np.ndarray,
    img2: np.ndarray,
    M: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    计算透视变换后的画布尺寸

    确定变换后图像所需的最小画布大小

    参数:
        img1: 第一张图像
        img2: 第二张图像
        M: 单应性矩阵

    返回:
        x_min: 最小 X 坐标（负值表示需要扩展）
        y_min: 最小 Y 坐标
        x_max: 最大 X 坐标
        y_max: 最大 Y 坐标
    """
    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 定义角点
    corners1 = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # 变换第二张图像的角点
    corners2_transformed = cv.perspectiveTransform(corners2, M)

    # 合并所有角点
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)

    # 计算边界
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - CANVAS_PADDING)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + CANVAS_PADDING)

    return x_min, y_min, x_max, y_max


def align_images(
    img1: np.ndarray,
    img2: np.ndarray,
    M: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    使用单应性矩阵对齐两张图像

    将 img2 变换到 img1 的坐标系中

    参数:
        img1: 参考图像（保持不动）
        img2: 待变换图像
        M: 从 img2 到 img1 的单应性矩阵

    返回:
        aligned_img2: 变换后与 img1 对齐的图像
        img1: 原始参考图像
        offset: (offset_x, offset_y) 变换后的偏移量
    """
    # 获取参考图像尺寸
    h1, w1 = img1.shape[:2]

    # 计算变换所需画布
    x_min, y_min, x_max, y_max = get_canvas_size(img1, img2, M)

    # 计算平移量
    offset_x = -x_min
    offset_y = -y_min
    offset = (offset_x, offset_y)

    # 构建平移矩阵
    translation_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # 组合变换：先单应性，再平移
    combined_matrix = translation_matrix @ M

    # 执行变换
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    aligned_img2 = cv.warpPerspective(
        img2,
        combined_matrix,
        (canvas_width, canvas_height)
    )

    # 在画布上放置参考图像
    # 创建扩展后的参考图像画布
    img1_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=img1.dtype)
    img1_canvas[offset_y:offset_y + h1, offset_x:offset_x + w1] = img1

    return aligned_img2, img1_canvas, offset


# ============== 辅助函数 ==============

@jit(nopython=True)
def get_w(wrap_img: np.ndarray) -> Tuple[int, int]:
    """
    检测图像中的有效重叠列范围（使用 JIT 加速）

    通过检测每列是否全为零像素（黑色/无效区域）来确定重叠区域

    参数:
        wrap_img: 输入图像，通常是变换后的候选图像
                  无效区域为纯黑色像素 [0, 0, 0]

    返回:
        case 1: (0, b) - 当从左侧开始检测到无效列时
                b 是最后一个有效列的索引
        case 2: (a, wrap_img.shape[1] - 1) - 当从右侧检测到无效列时
                a 是第一个有效列的索引
        case 3: (0, wrap_img.shape[1]) - 当没有检测到无效列时
                整个图像宽度都是有效区域

    检测逻辑：
    1. 从左向右扫描，查找第一列非全黑的位置
    2. 如果扫描到最右侧都没找到，说明图像从右侧开始无效
    3. 如果找到有效列，尝试从右向左扫描查找右侧边界

    注意:
        - 使用 @jit(nopython=True) 装饰器进行 JIT 编译
        - nopython=True 模式下代码不能使用 Python 解释器功能
        - 只能使用 NumPy 和 Python 基本语法
    """
    # 获取图像的行数和列数
    # shape[0] 是行数（高度），shape[1] 是列数（宽度）
    col = wrap_img.shape[0]  # 行数（高度）
    row = wrap_img.shape[1]  # 列数（宽度）

    # 标志变量
    b = -1  # 右侧边界索引，-1 表示未找到
    a = -1  # 左侧边界索引，-1 表示未找到
    flag = False   # 左侧扫描标志：True 表示当前列全黑
    flag1 = False  # 右侧扫描标志：True 表示当前列全黑

    # ========== 第一步：从左向右扫描 ==========
    # 目的：检测图像左侧是否存在无效（全黑）区域
    for i in range(wrap_img.shape[1]):  # 遍历每一列
        for j in range(wrap_img.shape[0]):  # 遍历每一行
            # 检查当前像素是否为全黑 [0, 0, 0]
            # BGR 三个通道都为 0 表示无效像素
            if (wrap_img[j][i][0] == 0 and
                wrap_img[j][i][1] == 0 and
                wrap_img[j][i][2] == 0):
                flag = True  # 当前像素是黑色，继续检查
            else:
                flag = False  # 发现非黑色像素，当前列有效
                break  # 停止检查当前列

        if flag:
            # 整列都是黑色，记录列索引
            b = i
            break  # 找到第一个全黑列，停止扫描

    # ========== 第二步：判断检测结果 ==========

    # 情况 1：找到了左侧的全黑列，且不是第 0 列
    if b != -1 and b != 0:
        # 返回从第 0 列到 b 列的范围
        # 表示有效区域在左侧
        return 0, b

    # 情况 2：第 0 列就是全黑的（图像从右侧开始有效）
    elif b == 0:
        # 从右向左扫描，查找左侧边界
        for i in range(wrap_img.shape[1] - 1, -1, -1):  # 从最后一列往前扫描
            for j in range(wrap_img.shape[0]):  # 遍历每一行
                # 检查当前像素是否为全黑
                if (wrap_img[j][i][0] == 0 and
                    wrap_img[j][i][1] == 0 and
                    wrap_img[j][i][2] == 0):
                    flag1 = True  # 当前像素是黑色
                else:
                    flag1 = False  # 发现非黑色像素
                    break

            if flag1:
                # 找到全黑列，记录为左侧边界
                a = i
                # 返回从 a 到最后一列的范围
                return a, wrap_img.shape[1] - 1

        # 如果反向扫描也没找到有效边界，返回整个图像宽度
        return 0, wrap_img.shape[1] - 1

    # 情况 3：没有找到任何全黑列（整个图像都有效）
    elif b == -1:
        # 返回整个图像的列范围
        return 0, wrap_img.shape[1]


def validate_homography(
    M: np.ndarray,
    img1_shape: Tuple[int, int],
    img2_shape: Tuple[int, int]
) -> bool:
    """
    验证单应性矩阵的合理性

    检查单应性矩阵是否会导致不合理的变换

    参数:
        M: 3x3 单应性矩阵
        img1_shape: 第一张图像的形状 (height, width)
        img2_shape: 第二张图像的形状 (height, width)

    返回:
        True 如果单应性矩阵合理，False 否则
    """
    # 检查矩阵是否为奇异矩阵
    if np.abs(np.linalg.det(M)) < 1e-6:
        return False

    # 检查变换后的角点是否在合理范围内
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]

    corners = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    transformed = cv.perspectiveTransform(corners, M)

    # 检查变换后的点是否导致极端的畸变
    max_dist = np.max(np.abs(transformed[:, 0, 0])) + np.max(np.abs(transformed[:, 0, 1]))

    if max_dist > 10 * max(h1, w1):  # 变换后距离超过原始尺寸的 10 倍
        return False

    return True


def estimate_overlap_ratio(
    img1: np.ndarray,
    img2: np.ndarray,
    M: np.ndarray
) -> float:
    """
    估计两张图像的重叠比例

    参数:
        img1: 第一张图像
        img2: 第二张图像
        M: 单应性矩阵

    返回:
        重叠比例 (0.0 到 1.0)
    """
    h1, w1 = img1.shape[:2]

    # 变换 img2 的角点
    corners2 = np.float32([[0, 0], [0, img2.shape[1]], [img2.shape[:2]], [img2.shape[0], 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv.perspectiveTransform(corners2, M)

    # 计算重叠区域（简化估计）
    # 假设重叠主要发生在图像边缘附近
    x_coords = corners2_transformed[:, 0, 0]
    overlap_width = min(w1, np.max(x_coords) - np.min(x_coords))

    return overlap_width / w1


# ============== 模块入口（用于测试）=============

if __name__ == "__main__":
    # 模块测试代码
    print("图像拼接几何变换模块")
    print("使用方法：from stitch.stitching_function import get_warp, find_overlap_region")
