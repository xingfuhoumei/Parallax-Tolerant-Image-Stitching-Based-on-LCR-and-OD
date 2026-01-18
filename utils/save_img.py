# -*- coding: utf-8 -*-
"""
图像保存模块
===========
功能说明：
1. 保存拼接管线的各个阶段结果图像
2. 按照命名规范组织输出文件
3. 自动创建输出目录

作者：gyd
日期：2023
"""

import os  # 操作系统接口
from pathlib import Path  # 路径操作
from typing import Optional  # 类型注解

import cv2  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库

# 从配置模块导入目录创建函数
from config import ensure_dir


# ============== 图像保存配置 ==============

# JPEG 图像保存质量参数
JPEG_QUALITY = 95  # 1-100，值越大质量越高


# ============== 主要保存函数 ==============

def SaveImage(
    img1: np.ndarray,
    img2: np.ndarray,
    path: str,
    H4matches: np.ndarray,
    warp_img1: np.ndarray,
    overlap: np.ndarray,
    overlap1: np.ndarray,
    seam_output: np.ndarray,
    H4stitching: np.ndarray,
    matches1: Optional[np.ndarray] = None
) -> str:
    """
    保存拼接管线的所有输出图像

    该函数会保存管线各个阶段的中间结果，便于调试和结果展示

    参数:
        img1: 第一张输入图像（候选图像）
        img2: 第二张输入图像（参考图像）
        path: 输出目录路径
        H4matches: 提议方法的匹配点可视化图像
        warp_img1: 透视变换后的图像
        overlap: 重叠区域融合结果
        overlap1: 带接缝的重叠区域可视化
        seam_output: 接缝输出结果
        H4stitching: 最终拼接结果
        matches1: RANSAC 匹配点可视化图像（可选）

    返回:
        输出目录的完整路径

    文件命名:
        1-candidate.jpg     - 候选图像
        2-reference.jpg     - 参考图像
        3-ransac_mp.jpg     - RANSAC 匹配点可视化
        4-proposed_mp.jpg   - 提议方法匹配点可视化
        5-overlap_energy.jpg - 重叠区域能量图
        6-seam_output.jpg   - 接缝输出可视化
        7-overlap.jpg       - 重叠区域结果
        8-H4stitching.jpg   - 最终拼接结果
    """
    # 如果没有文件夹自动创造文件夹
    if not os.path.exists(path):
        os.makedirs(path)

    # 确保路径拼接始终带分隔符，避免文件写到上级目录
    def _p(filename):
        # os.path.join 会自动添加路径分隔符
        return os.path.join(path, filename)

    # 依次保存每个阶段的可视化结果

    # 1. 保存候选图像
    cv2.imwrite(_p('1-candidate.jpg'), img1)

    # 2. 保存参考图像
    cv2.imwrite(_p('2-reference.jpg'), img2)

    # 3. 保存 RANSAC 匹配点可视化（如果存在）
    # 匹配图可能为空，做保护
    if matches1 is not None:
        cv2.imwrite(_p('3-ransac_mp.jpg'), matches1)

    # 4. 保存提议方法匹配点可视化（如果存在）
    if H4matches is not None:
        cv2.imwrite(_p('4-proposed_mp.jpg'), H4matches)

    # 5. 保存重叠区域能量图
    cv2.imwrite(_p('5-overlap_energy.jpg'), overlap1)

    # 6. 保存接缝输出可视化
    cv2.imwrite(_p('6-seam_output.jpg'), seam_output)

    # 7. 保存重叠区域结果
    cv2.imwrite(_p('7-overlap.jpg'), overlap)

    # 8. 保存最终拼接结果
    cv2.imwrite(_p('8-H4stitching.jpg'), H4stitching)

    print("已成功保存图片到输出文件夹")
    return ""


def ShowImage(
    matches1: Optional[np.ndarray],
    H4matches: np.ndarray,
    warp_img1: np.ndarray,
    overlap: np.ndarray,
    overlap1: np.ndarray,
    seam_output: np.ndarray,
    H4stitching: np.ndarray
) -> None:
    """
    显示拼接管线的各个阶段结果图像

    该函数会创建多个窗口显示中间结果，便于可视化调试

    参数:
        matches1: RANSAC 匹配点可视化图像（可选）
        H4matches: 提议方法的匹配点可视化图像
        warp_img1: 透视变换后的图像
        overlap: 重叠区域融合结果
        overlap1: 带接缝的重叠区域可视化
        seam_output: 接缝输出结果
        H4stitching: 最终拼接结果
    """
    # 显示 RANSAC 匹配点（如果存在）
    if matches1 is not None:
        cv2.imshow("RANSAC_points", matches1)

    # 显示各个阶段的结果
    cv2.imshow('Proposed_stitching', H4matches)
    cv2.imshow('warp_img1', warp_img1)
    cv2.imshow('overlap', overlap)
    cv2.imshow('overlap1', overlap1)
    cv2.imshow('seam_output', seam_output)
    cv2.imshow('H4stitching', H4stitching)

    # 等待按键关闭所有窗口
    cv2.waitKey()
    print("图都放完啦,瞅你咋💧")


def save_image(
    image: np.ndarray,
    filepath: str,
    quality: int = JPEG_QUALITY
) -> bool:
    """
    保存单张图像到指定路径

    参数:
        image: 要保存的图像（NumPy 数组）
        filepath: 保存路径
        quality: JPEG 质量 (1-100)

    返回:
        保存是否成功
    """
    try:
        # 确保父目录存在
        ensure_dir(Path(filepath).parent)

        # 根据 filepath 后缀选择保存方式
        ext = Path(filepath).suffix.lower()

        if ext in ['.jpg', '.jpeg']:
            # JPEG 格式：可指定质量，压缩比高
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            # PNG 格式：无损压缩
            cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            # 其他格式：使用默认参数
            cv2.imwrite(filepath, image)

        return True

    except Exception as e:
        print(f"保存图像失败 {filepath}: {e}")
        return False


def save_debug_image(
    image: np.ndarray,
    name: str,
    output_dir: str
) -> str:
    """
    保存调试图像

    用于保存管线中间结果的调试图像

    参数:
        image: 要保存的图像
        name: 图像名称（不含扩展名）
        output_dir: 输出目录

    返回:
        保存的文件完整路径
    """
    # 创建调试目录
    debug_dir = ensure_dir(output_dir)

    # 构建文件路径
    filepath = os.path.join(debug_dir, f"{name}.jpg")

    # 保存图像
    save_image(image, filepath)

    return filepath
