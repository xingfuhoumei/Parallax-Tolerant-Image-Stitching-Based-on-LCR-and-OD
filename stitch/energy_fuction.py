# -*- coding: utf-8 -*-
"""
能量函数计算模块
===========
功能说明：
1. 计算图像匹配的能量代价
2. 差分图像边缘检测
3. 匹配点高斯权重计算
4. 视差掩码融合

主要算法：
- 图像差分与边缘权重
- 基于匹配点距离的高斯权重
- 多权重融合策略
- Min-Max 归一化

作者：gyd
日期：2023
"""

from typing import Tuple  # 类型注解

import cv2 as cv  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库
from numba import jit  # JIT 编译加速
from sklearn import preprocessing  # 数据预处理与归一化


# ============== 配置常量 ==============

# 高斯核的标准差参数（相对于图像尺寸）
SIGMA_RATIO = 0.05  # sigma = 0.05 * min(h, w)

# 权重融合系数（当使用视差掩码时）
WEIGHT_DIFF_PARALLAX = 0.6  # 差分权重
WEIGHT_GAUSSIAN_PARALLAX = 0.1  # 高斯权重
WEIGHT_PARALLAX_MASK = 0.3  # 视差掩码权重

# 权重融合系数（不使用视差掩码时）
WEIGHT_DIFF_NORMAL = 0.4  # 差分权重
WEIGHT_GAUSSIAN_NORMAL = 0.6  # 高斯权重


# ============== 主要功能函数 ==============

def get_cost(
    result_img1: np.ndarray,
    result_img2: np.ndarray,
    dst_points: np.ndarray,
    dst_min_overlap_point_x: int,
    src_after: np.ndarray,
    parallax_mask: np.ndarray,
    use_parallax: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算图分割的能量代价函数

    该函数综合多种因素计算每个像素的能量值：
    1. 图像差分权重（边缘检测）
    2. 基于匹配点距离的高斯权重
    3. （可选）视差掩码权重

    参数:
        result_img1: 第一张对齐后的图像
        result_img2: 第二张对齐后的图像
        dst_points: 目标匹配点坐标数组
        dst_min_overlap_point_x: 重叠区域的最小 X 坐标
        src_after: 变换后的源匹配点坐标
        parallax_mask: 视差掩码（可选）
        use_parallax: 是否使用视差掩码

    返回:
        weight: 浮点型能量权重矩阵 (dtype=float64)
        weight1: uint8 型能量权重矩阵，范围 [0, 255]
    """
    # ========== 步骤1：计算匹配点之间的距离分数 ==========
    # 这个分数反映了每个匹配对的几何一致性
    matches_scores = get_matches_scores(dst_points, src_after)

    # ========== 步骤2：调整目标点坐标 ==========
    # 将 X 坐标平移，使其相对于重叠区域起点
    # 这样后续计算时坐标从 0 开始
    dst_points[:, 0] = dst_points[:, 0] - dst_min_overlap_point_x

    # ========== 步骤3：初始化权重矩阵 ==========
    # 获取重叠区域的高度和宽度
    h, w = result_img1.shape[0], result_img1.shape[1]

    # 创建两个权重矩阵用于累加
    # distances: 最终的权重矩阵
    # dis_tmp: 临时权重矩阵（每次迭代的中间结果）
    distances = np.zeros((h, w), dtype=np.float64)
    dis_tmp = np.zeros((h, w), dtype=np.float64)

    # ========== 步骤4：设置高斯模糊参数 ==========
    # gamma: 控制高斯函数的衰减速度
    # sigma: 控制高斯核的宽度（与图像尺寸成正比）
    gama = 0.01
    sigma = SIGMA_RATIO * min(h, w)

    # ========== 步骤5：计算图像差分权重 ==========
    # 使用 cv2.subtract 进行差分运算
    # 这会保留正值，负值被截断为 0
    edge_map1 = cv.subtract(result_img1, result_img2)  # img1 - img2
    edge_map2 = cv.subtract(result_img2, result_img1)  # img2 - img1

    # 合并两个差分图，得到总的差异图
    # 在有差异的地方会有较高的值
    edge = edge_map1 + edge_map2

    # 转换为灰度图作为差分权重
    # COLOR_BGR2GRAY 将 BGR 图像转换为单通道灰度图
    diff_weight = cv.cvtColor(edge, cv.COLOR_BGR2GRAY)

    # ========== 步骤6：计算基于匹配点的高斯权重 ==========
    # 这个权重反映了每个像素距离匹配点的远近
    # 距离匹配点越近，权重越高
    points_weight = get_point_weight(
        dst_points,      # 匹配点坐标
        h,               # 高度
        w,               # 宽度
        dis_tmp,         # 临时矩阵
        distances,       # 累加矩阵
        matches_scores   # 匹配分数
    )

    # 对点权重矩阵应用高斯模糊，使权重分布更平滑
    Gaussian_weight = cv.GaussianBlur(points_weight, (3, 3), sigmaX=0)

    # ========== 步骤7：归一化高斯权重到 [0, 255] 范围 ==========
    # MinMaxScaler 将数据线性映射到指定范围
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))

    # fit_transform 先拟合数据（计算最小值和最大值），然后进行变换
    normalize_Gaussian_weight = minmax_scaler.fit_transform(Gaussian_weight)

    # 转换为 uint8 类型，范围 [0, 255]
    normalize_Gaussian_weight1 = np.array(normalize_Gaussian_weight, dtype='uint8')

    # ========== 步骤8：融合多种权重 ==========
    if use_parallax:
        # 使用视差掩码的情况：三种权重融合
        # 将视差掩码转换为灰度图
        img_parallax1 = cv.cvtColor(parallax_mask, cv.COLOR_RGB2GRAY)

        # 注意：代码中有两行权重计算，最后一行会覆盖前面的
        # 最终使用的是：20% 差分 + 80% 高斯 + 0% 视差掩码
        weight = (0.2 * diff_weight +
                  0.8 * normalize_Gaussian_weight +
                  0.0 * img_parallax1)
    else:
        # 不使用视差掩码的情况：两种权重融合
        weight = (0.4 * diff_weight +
                  0.6 * normalize_Gaussian_weight)

    # ========== 步骤9：设置边界约束 ==========
    # 将左右两列的权重设置为最大值（255）
    # 这确保在图分割时接缝会从边界开始或结束
    weight[:, 0] = 255   # 左边界
    weight[:, -1] = 255  # 右边界

    # ========== 步骤10：转换为 uint8 类型 ==========
    weight1 = np.array(weight, dtype='uint8')

    return weight, weight1


# ============== JIT 加速的辅助函数 ==============

@jit(nopython=True)
def get_matches_scores(
    dst: np.ndarray,
    src_after: np.ndarray
) -> np.ndarray:
    """
    计算匹配点对之间的欧几里得距离分数（使用 JIT 加速）

    对于每对匹配点，计算它们在变换前后的距离差异

    参数:
        dst: 目标点坐标数组，形状 (N, 2)
             每行是 [x, y] 坐标
        src_after: 变换后的源点坐标数组，形状 (N, 2)
                   每行是 [x, y] 坐标

    返回:
        score: 距离分数数组，形状 (1, N)
               每个元素是对应点对的欧几里得距离

    计算公式:
        distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    """
    # 初始化分数数组
    # 形状为 (1, N)，N 是匹配点的数量
    score = np.zeros((1, dst.shape[0]))

    # 遍历每一对匹配点
    for i in range(dst.shape[0]):
        # 提取坐标
        x_dst = dst[i][0]   # 目标点的 X 坐标
        y_dst = dst[i][1]   # 目标点的 Y 坐标
        x_src = src_after[i][0]  # 变换后源点的 X 坐标
        y_src = src_after[i][1]  # 变换后源点的 Y 坐标

        # 计算坐标差值的平方
        dx_squared = (x_dst - x_src) ** 2
        dy_squared = (y_dst - y_src) ** 2

        # 计算欧几里得距离（平方根）
        distance = (dx_squared + dy_squared) ** 0.5

        # 存储距离分数
        score[0][i] = distance

    return score


@jit(nopython=True)
def get_point_weight(
    dst: np.ndarray,
    h: int,
    w: int,
    dis_tmp: np.ndarray,
    distances: np.ndarray,
    matches_scores: np.ndarray
) -> np.ndarray:
    """
    基于匹配点计算高斯权重图（使用 JIT 加速）

    对于每个匹配点，计算它对所有像素的高斯权重贡献
    距离匹配点越近的像素，获得越高的权重

    参数:
        dst: 匹配点坐标数组，形状 (N, 2)
             每行是 [x, y] 或 [col, row] 坐标
        h: 图像高度（行数）
        w: 图像宽度（列数）
        dis_tmp: 临时权重矩阵，用于单点计算
        distances: 累加权重矩阵，存储所有点的贡献
        matches_scores: 每个匹配点的分数数组，形状 (1, N)

    返回:
        distances: 累加后的权重矩阵，形状 (h, w)

    算法说明:
        对于每个匹配点 (x, y):
        1. 计算该点到所有像素 (row, col) 的距离
        2. 距离乘以匹配分数作为权重
        3. 累加到总权重矩阵中
    """
    # 遍历每个匹配点
    for i in range(dst.shape[0]):
        # 提取匹配点的坐标
        # dst[i][0] 是列坐标（X），dst[i][1] 是行坐标（Y）
        x = dst[i][0]  # 列坐标
        y = dst[i][1]  # 行坐标

        # ========== 计算该匹配点对所有像素的权重贡献 ==========
        # 遍历所有行
        for row in range(h):
            # 遍历所有列
            for col in range(w):
                # 计算像素 (row, col) 到匹配点 (y, x) 的欧几里得距离
                # 注意：row 是行索引对应 Y，col 是列索引对应 X
                dx = x - col  # X 方向距离
                dy = y - row  # Y 方向距离
                pixel_distance = (dx ** 2 + dy ** 2) ** 0.5  # 欧几里得距离

                # 计算权重：距离 × 匹配分数
                # matches_scores[0][i] 是第 i 个匹配点的分数
                # 距离越近或匹配分数越高，权重贡献越小（因为要在能量函数中使用）
                weight_contribution = matches_scores[0][i] * pixel_distance

                # 存储到临时矩阵
                dis_tmp[row][col] = weight_contribution

        # ========== 累加到总权重矩阵 ==========
        # 将当前匹配点的贡献加到总权重中
        distances = distances + dis_tmp

    return distances


# ============== 高级权重计算函数 ==============

def compute_edge_weight(
    img1: np.ndarray,
    img2: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    计算基于边缘检测的权重图

    使用 Sobel 算子检测边缘，在有强边缘的地方给予更高权重

    参数:
        img1: 第一张图像
        img2: 第二张图像
        kernel_size: Sobel 核大小

    返回:
        edge_weight: 边缘权重图，范围 [0, 255]
    """
    # 转换为灰度图
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 计算 Sobel 梯度
    grad_x1 = cv.Sobel(gray1, cv.CV_64F, 1, 0, ksize=kernel_size)
    grad_y1 = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=kernel_size)
    grad_x2 = cv.Sobel(gray2, cv.CV_64F, 1, 0, ksize=kernel_size)
    grad_y2 = cv.Sobel(gray2, cv.CV_64F, 0, 1, ksize=kernel_size)

    # 计算梯度幅值
    magnitude1 = np.sqrt(grad_x1**2 + grad_y1**2)
    magnitude2 = np.sqrt(grad_x2**2 + grad_y2**2)

    # 计算梯度差异
    edge_diff = np.abs(magnitude1 - magnitude2)

    # 归一化到 [0, 255]
    edge_weight = cv.normalize(edge_diff, None, 0, 255, cv.NORM_MINMAX)

    return edge_weight.astype(np.uint8)


def compute_gradient_weight(
    img1: np.ndarray,
    img2: np.ndarray
) -> np.ndarray:
    """
    计算基于梯度方向的权重图

    检测两张图像的梯度方向差异，在方向不一致的地方给予更高权重

    参数:
        img1: 第一张图像
        img2: 第二张图像

    返回:
        gradient_weight: 梯度方向权重图，范围 [0, 255]
    """
    # 转换为灰度图
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 计算 Sobel 梯度
    grad_x1 = cv.Sobel(gray1, cv.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=3)
    grad_x2 = cv.Sobel(gray2, cv.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv.Sobel(gray2, cv.CV_64F, 0, 1, ksize=3)

    # 计算梯度方向（角度）
    angle1 = np.arctan2(grad_y1, grad_x1)
    angle2 = np.arctan2(grad_y2, grad_x2)

    # 计算角度差异（处理周期性）
    angle_diff = np.abs(angle1 - angle2)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

    # 归一化到 [0, 255]
    gradient_weight = cv.normalize(angle_diff, None, 0, 255, cv.NORM_MINMAX)

    return gradient_weight.astype(np.uint8)


def normalize_weight(
    weight: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    归一化权重矩阵

    参数:
        weight: 输入权重矩阵
        method: 归一化方法
                - 'minmax': Min-Max 归一化到 [0, 255]
                - 'zscore': Z-Score 标准化后截断到 [0, 255]

    返回:
        normalized: 归一化后的权重矩阵，dtype=uint8
    """
    if method == 'minmax':
        # Min-Max 归一化
        normalized = cv.normalize(weight, None, 0, 255, cv.NORM_MINMAX)
    elif method == 'zscore':
        # Z-Score 标准化
        mean = np.mean(weight)
        std = np.std(weight)
        if std > 0:
            normalized = (weight - mean) / std
            # 截断到 [-3, 3] 范围（99.7% 的数据）
            normalized = np.clip(normalized, -3, 3)
            # 映射到 [0, 255]
            normalized = ((normalized + 3) / 6) * 255
        else:
            normalized = np.zeros_like(weight)
        normalized = normalized.astype(np.uint8)
    else:
        raise ValueError(f"未知的归一化方法: {method}")

    return normalized


# ============== 模块入口（用于测试）=============

if __name__ == "__main__":
    # 模块测试代码
    print("能量函数计算模块")
    print("使用方法：from stitch.energy_fuction import get_cost")
