# -*- coding: utf-8 -*-
"""
SAM 视差检测模块
===========
功能说明：
1. 使用 Segment Anything Model (SAM) 进行视差区域分割
2. 支持点提示 (point prompts) 进行交互式分割
3. 自动检测视差区域并生成掩码

作者：gyd
日期：2023

依赖:
    - segment_anything: SAM 模型
    - numpy: 数值计算
    - opencv-python: 图像处理
    - matplotlib: 可视化
"""

import sys  # 系统模块
from pathlib import Path  # 路径操作
from typing import Tuple, Optional  # 类型注解

import cv2  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库
from matplotlib import pyplot as plt  # 绘图库

# 导入 SAM 相关模块
from segment_anything import sam_model_registry, SamPredictor

# 导入配置
from config import SAM_CHECKPOINT


# ============== SAM 配置参数 ==============

# SAM 模型类型
# vit_b: ViT-B 模型（基础版本）
# vit_l: ViT-L 模型（大型版本）
# vit_h: ViT-H 模型（超大型版本）
MODEL_TYPE = "vit_b"

# 设备配置：使用 GPU 进行推理，如果不可用则使用 CPU
DEVICE = "cuda"  # 可选: "cpu"


# ============== 辅助可视化函数 ==============

def show_mask(
    mask: np.ndarray,
    ax,
    random_color: bool = False
) -> None:
    """
    在指定坐标轴上显示掩码

    参数:
        mask: 要显示的掩码 (H x W) 或 (H x W x 1)
        ax: matplotlib 坐标轴对象
        random_color: 是否使用随机颜色
    """
    # 根据参数选择颜色
    if random_color:
        # 生成随机颜色
        color = np.concatenate([
            np.random.random(3),  # RGB 三个随机值
            np.array([0.6])  # Alpha 通道固定为 0.6
        ], axis=0)
    else:
        # 使用固定的白色（半透明）
        color = np.array([
            255 / 255,  # R
            255 / 255,  # G
            255 / 255,  # B
            1          # Alpha
        ])

    # 获取掩码的高度和宽度
    h, w = mask.shape[-2:]

    # 调整掩码形状以匹配颜色
    mask_reshaped = mask.reshape(h, w, 1)

    # 创建颜色掩码并显示
    mask_image = mask_reshaped * color.reshape(1, 1, -1)

    # 在坐标轴上显示掩码
    ax.imshow(mask_image)


def show_points(
    coords: np.ndarray,
    labels: np.ndarray,
    ax,
    marker_size: int = 375
) -> None:
    """
    在指定坐标轴上显示点

    参数:
        coords: 点坐标数组 (N x 2)
        labels: 点标签数组 (N,)，1 表示前景点，0 表示背景点
        ax: matplotlib 坐标轴对象
        marker_size: 标记大小
    """
    # 分离前景点和背景点
    pos_points = coords[labels == 1]  # 前景点（绿色）
    neg_points = coords[labels == 0]  # 背景点（红色）

    # 绘制前景点（绿色星号）
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color='green',
        marker='*',
        s=marker_size,
        edgecolor='white',
        linewidth=1.25
    )

    # 绘制背景点（红色星号）
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color='red',
        marker='*',
        s=marker_size,
        edgecolor='white',
        linewidth=1.25
    )


def show_box(
    box: np.ndarray,
    ax
) -> None:
    """
    在指定坐标轴上显示边界框

    参数:
        box: 边界框 [x1, y1, x2, y2]
        ax: matplotlib 坐标轴对象
    """
    # 提取边界框坐标
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # 创建并绘制矩形
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),  # 左上角坐标
            w,           # 宽度
            h,           # 高度
            edgecolor='green',
            facecolor=(0, 0, 0, 0),  # 填充色为透明
            lw=2          # 线宽
        )
    )


# ============== 主要功能函数 ==============

def find_parallax(
    image: np.ndarray,
    matrix: np.ndarray
) -> np.ndarray:
    """
    使用 SAM 检测视差区域

    该函数：
    1. 加载 SAM 模型
    2. 对输入图像进行编码
    3. 根据输入点坐标计算中心点
    4. 使用点提示进行分割
    5. 返回视差区域掩码

    参数:
        image: 输入图像（BGR 格式）
        matrix: 视差点坐标数组 (N x 2)

    返回:
        mask_image: 生成的视差区域掩码图像
    """
    # 确保 matrix 是 numpy 数组
    matrix = np.array(matrix)

    # 检查矩阵是否为空或形状不正确
    if matrix.size == 0 or len(matrix.shape) < 2 or matrix.shape[0] == 0:
        # 返回空的黑色掩码
        return np.zeros((image.shape[0], image.shape[1], 4))

    # 转换 BGR 到 RGB（SAM 需要 RGB 输入）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载 SAM 模型
    print(f"正在加载 SAM 模型: {SAM_CHECKPOINT}")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)

    # 创建预测器
    predictor = SamPredictor(sam)

    # 对图像进行编码（预处理）
    predictor.set_image(image)

    # 计算视差点的中心坐标作为提示点
    # 取所有点的平均值作为中心
    input_point = np.array([np.mean(matrix[:, 0]), np.mean(matrix[:, 1])]).reshape(1, 2)

    # 设置提示标签：1 表示前景点
    input_label = np.array([1])

    # 使用 SAM 进行预测，获取多个掩码
    # multimask_output=True 返回多个掩码结果
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # 选择最佳掩码（分数最高的）
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]

    # 将掩码转换为 RGBA 格式的图像数组返回
    # 创建 RGBA 图像（R=mask, G=mask, B=mask, A=mask）
    mask_rgba = np.zeros((best_mask.shape[0], best_mask.shape[1], 4), dtype=np.float32)
    mask_rgba[:, :, 0] = best_mask  # R
    mask_rgba[:, :, 1] = best_mask  # G
    mask_rgba[:, :, 2] = best_mask  # B
    mask_rgba[:, :, 3] = best_mask  # A

    return mask_rgba


def find_parallax_batch(
    image: np.ndarray,
    points_list: list,
    device: str = DEVICE
) -> list:
    """
    批量处理多个视差区域

    参数:
        image: 输入图像
        points_list: 视差点坐标列表，每个元素是一个 (N x 2) 数组
        device: 计算设备

    返回:
        masks_list: 每个视差区域的掩码列表
    """
    # 转换 BGR 到 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载 SAM 模型
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)

    # 创建预测器
    predictor = SamPredictor(sam)

    # 对图像进行编码
    predictor.set_image(image_rgb)

    # 存储所有掩码
    masks_list = []

    # 处理每个视差点集
    for points in points_list:
        # 计算中心点
        center = np.mean(points, axis=0).reshape(1, 2)
        input_label = np.array([1])

        # 预测
        masks, scores, _ = predictor.predict(
            point_coords=center,
            point_labels=input_label,
            multimask_output=True,
        )

        # 选择最佳掩码
        best_mask = masks[np.argmax(scores)]

        # 将掩码转换为 uint8
        mask_uint8 = (best_mask * 255).astype(np.uint8)

        # 如果是 RGBA，转换为 RGB
        if mask_uint8.shape[-1] == 4:
            mask_uint8 = mask_uint8[:, :, :3]

        masks_list.append(mask_uint8)

    return masks_list


def refine_mask_with_morphology(
    mask: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    使用形态学操作优化掩码

    参数:
        mask: 输入掩码
        kernel_size: 形态学核大小

    返回:
        refined_mask: 优化后的掩码
    """
    # 确保是 uint8 类型
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # 如果是 RGBA，转换为 RGB
    if mask.shape[-1] == 4:
        mask_rgb = mask[:, :, :3]
    else:
        mask_rgb = mask.copy()

    # 转换为灰度图
    if len(mask_rgb.shape) == 3:
        gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask_rgb.copy()

    # 形态学闭运算（填充孔洞）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊
    refined = cv2.GaussianBlur(closed, (5, 5), 0)

    return refined


# ============== 可选的交互式分割功能 ==============

def interactive_segmentation(
    image_path: str,
    checkpoint_path: Optional[str] = None
) -> None:
    """
    交互式视差区域分割工具

    允许用户通过点击图像选择视差区域进行分割

    参数:
        image_path: 输入图像路径
        checkpoint_path: SAM 模型检查点路径，如果为 None 则使用默认配置
    """
    # 设置检查点路径
    if checkpoint_path is None:
        checkpoint_path = str(SAM_CHECKPOINT)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载 SAM 模型
    print(f"加载 SAM 模型: {checkpoint_path}")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)

    # 创建预测器
    predictor = SamPredictor(sam)

    # 对图像进行编码
    predictor.set_image(image_rgb)

    # 设置鼠标回调函数
    def on_mouse_click(event, x, y, flags, param):
        """处理鼠标点击事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 将点击坐标转换为 numpy 数组
            input_point = np.array([[x, y]])
            input_label = np.array([1])  # 前景点

            # 进行预测
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # 获取最佳掩码
            best_mask = masks[np.argmax(scores)]

            # 在原图上显示掩码
            display_image = image_rgb.copy()
            display_image[best_mask] = display_image[best_mask] * 0.5

            plt.figure(figsize=(10, 10))
            plt.imshow(display_image)
            plt.title(f"Score: {scores[np.argmax(scores)]:.3f}")
            plt.axis('on')
            plt.draw()
            plt.show()

    # 显示图像并设置回调
    cv2.imshow('Image', image_rgb)
    cv2.setMouseCallback('Image', on_mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
