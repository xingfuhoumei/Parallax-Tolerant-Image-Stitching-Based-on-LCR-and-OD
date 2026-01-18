# -*- coding: utf-8 -*-
"""
视差区域处理模块
===========
功能说明：
1. 查找视差区域的边界
2. 生成视差区域的掩码
3. 使用 approxPolyDP 进行多边形近似

作者：gyd
日期：2023
"""

from typing import Tuple, List  # 类型注解

import cv2  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库


# ============== 主要功能函数 ==============

def find_area(img_desk: np.ndarray) -> np.ndarray:
    """
    在视差掩码图像中查找并生成区域掩码

    该函数会：
    1. 将图像转换为灰度图
    2. 找到非零像素的边界
    3. 生成填充边界的矩形区域掩码

    参数:
        img_desk: 输入的视差掩码图像（RGB 格式）

    返回:
        new: 生成的区域掩码图像，边界区域填充为白色
    """
    # 初始化 x 和 y 坐标列表
    x = []  # 行索引列表
    y = []  # 列索引列表

    # 读取图像并转换为灰度图
    # COLOR_RGB2GRAY 将 RGB 图像转换为灰度图
    img_gray = cv2.cvtColor(img_desk, cv2.COLOR_RGB2GRAY)

    # 查找非零像素的坐标范围
    a, b = find_points(img_gray, x, y)

    # 转换为 NumPy 数组
    a = np.array(a)  # 行坐标数组
    b = np.array(b)  # 列坐标数组

    # 打印边界信息
    print(f"视差区域边界: 行[{min(a)}:{max(a)}], 列[{min(b)}:{max(b)}]")

    # 创建与输入图像相同形状的全零数组
    new = np.zeros(img_desk.shape)

    # 在边界区域内填充白色
    # 从最小行到最大行
    for i in range(min(a), max(a) + 1):
        # 从最小列到最大列-20（留有一些边缘）
        for j in range(min(b), max(b) - 20):
            # 设置为白色 [255, 255, 255]
            new[i][j] = [255, 255, 255]

    # 转换为 uint8 类型
    new = np.array(new, dtype='uint8')

    return new


def find_points(
    img: np.ndarray,
    x: List[int],
    y: List[int]
) -> Tuple[List[int], List[int]]:
    """
    查找图像中非零像素的行和列索引

    该函数遍历图像，记录所有非零像素所在的行和列

    参数:
        img: 输入图像（通常是灰度图）
        x: 行索引列表（传入引用，会被修改）
        y: 列索引列表（传入引用，会被修改）

    返回:
        x: 包含非零像素的所有行索引
        y: 包含非零像素的所有列索引

    注意:
        x 和 y 是传入的列表引用，函数会直接修改它们
    """
    # 遍历图像的每一行
    for i in range(img.shape[0]):
        # 遍历图像的每一列
        for j in range(img.shape[1]):
            # 如果像素值不为零
            if img[i][j] != 0:
                # 如果行索引 i 还不在列表中，添加它
                if i not in x:
                    x.append(i)
                # 如果列索引 j 还不在列表中，添加它
                if j not in y:
                    y.append(j)

    return x, y


def create_parallax_mask(
    left_parallax: np.ndarray,
    right_parallax: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    创建组合的视差掩码

    将左右视差掩码合并为一个完整的掩码

    参数:
        left_parallax: 左侧视差掩码
        right_parallax: 右侧视差掩码
        width: 图像宽度
        height: 图像高度

    返回:
        combined_mask: 组合后的视差掩码
    """
    # 创建空白掩码
    combined_mask = np.zeros((height, width, 4), dtype=np.uint8)

    # 添加左侧视差（如果有）
    if left_parallax is not None and left_parallax.size > 0:
        # 转换 RGBA 到 RGB 并添加 alpha 通道
        if left_parallax.shape[-1] == 4:  # RGBA
            combined_mask += left_parallax
        else:  # RGB
            combined_mask[:, :, :3] += left_parallax

    # 添加右侧视差（如果有）
    if right_parallax is not None and right_parallax.size > 0:
        if right_parallax.shape[-1] == 4:  # RGBA
            combined_mask += right_parallax
        else:  # RGB
            combined_mask[:, :, :3] += right_parallax

    # 截断到有效范围 [0, 255]
    combined_mask = np.clip(combined_mask, 0, 255)

    return combined_mask


def refine_parallax_mask(
    mask: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    优化视差掩码，去除噪点

    使用形态学操作清理掩码中的小噪点

    参数:
        mask: 输入掩码
        kernel_size: 形态学操作核大小

    返回:
        refined_mask: 优化后的掩码
    """
    # 转换为灰度图
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask.copy()

    # 创建形态学操作核
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # 执行闭运算（先膨胀后腐蚀，填充小孔）
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 执行开运算（先腐蚀后膨胀，去除小点）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 高斯模糊平滑边缘
    blurred = cv2.GaussianBlur(opened, (5, 5), 0)

    return blurred


def get_parallax_center(
    mask: np.ndarray
) -> Tuple[int, int]:
    """
    获取视差区域的中心点坐标

    参数:
        mask: 视差掩码

    返回:
        center_x: 中心 x 坐标
        center_y: 中心 y 坐标
    """
    # 转换为灰度图
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask.copy()

    # 找到非零像素的坐标
    points = cv2.findNonZero(gray)

    # 计算中心点
    if len(points) > 0:
        # 使用图像矩计算质心
        moments = cv2.moments(points)
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
    else:
        # 如果没有找到点，返回图像中心
        center_x = mask.shape[1] // 2
        center_y = mask.shape[0] // 2

    return center_x, center_y
