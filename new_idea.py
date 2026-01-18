# -*- coding: utf-8 -*-
"""
图像拼接主入口模块
===========
功能说明：
这是 Samdup 图像拼接算法的主入口文件，实现了完整的拼接管线。

主要流程：
1. 特征提取与匹配 (SIFT)
2. RANSAC 单应性计算
3. 视差区域检测 (SAM)
4. 能量函数计算
5. 图割最优接缝查找
6. 图像融合与输出

创新点：
- 在 RANSAC 剔除的外点中检测视差区域
- 使用 SAM (Segment Anything Model) 进行精确的视差分割
- 结合能量函数与图割算法优化接缝位置

使用方法：
    python new_idea.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg --output ./output

作者：gyd
日期：2023
"""

import argparse  # 命令行参数解析
import sys  # 系统模块
import time  # 时间计时
from pathlib import Path  # 路径操作
from typing import Tuple, Optional, List  # 类型注解

import cv2 as cv  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库

# ============== 导入自定义模块 ==============

# 从配置模块导入默认值和工具函数
from config import (
    DEFAULT_NEW_IDEA_IMG1,    # 默认参考图像路径
    DEFAULT_NEW_IDEA_IMG2,    # 默认候选图像路径
    DEFAULT_NEW_IDEA_OUTPUT,  # 默认输出目录
    ensure_dir                # 确保目录存在的工具函数
)

# 特征匹配模块
from matches.Sift import get_initial_matches  # SIFT 特征提取与初始匹配
from matches.Ransac import get_RANSAC_points  # RANSAC 单应性计算
from matches.parallax_area import get_true_points  # 视差区域点提取

# 视差检测模块
from parallax.SAM import find_parallax  # SAM 视差区域检测
from parallax.find_approxPolyDP import find_area  # 视差区域边界查找

# 拼接模块
from stitch.energy_fuction import get_cost  # 能量函数计算
from stitch.graphcut import (
    graph_cut,            # 图割拼接
    get_stitched_image    # 拼接图像生成
)
from stitch.stitching_function import (
    get_warp,              # 透视变换与裁剪
    find_overlap_region,   # 重叠区域查找
)

# 工具模块
from utils.ToPrint import ToPrint  # 统计信息输出
from utils.save_img import SaveImage  # 图像保存

# 导入全局统计变量模块
import all  # 全局变量存储


# ============== 配置常量 ==============

# 图像处理默认参数
DEFAULT_SCALE = 0.5         # 默认图像缩放比例
DEFAULT_RATIO_TEST = 0.98   # 默认比例测试阈值

# 高斯模糊参数
GAUSSIAN_KERNEL_SIZE = 3    # 高斯核大小
GAUSSIAN_SIGMA = 0.4        # 高斯标准差


# ============== 终端输出美化工具 ==============

class TerminalColors:
    """终端颜色代码"""
    HEADER = '\033[95m'           # 亮粉色
    OK_BLUE = '\033[94m'          # 蓝色
    OK_GREEN = '\033[92m'        # 绿色
    WARNING = '\033[93m'         # 黄色
    ERROR = '\033[91m'           # 红色
    ENDC = '\033[0m'            # 结束颜色
    BOLD = '\033[1m'             # 粗体
    UNDERLINE = '\033[4m'         # 下划线


class ProgressBar:
    """简单的文本进度条"""

    def __init__(self, total: int, prefix: str = "处理中"):
        """
        初始化进度条

        参数:
            total: 总步数
            prefix: 进度条前缀文本
        """
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.start_time = time.time()

    def update(self, step: int = 1, description: str = ""):
        """
        更新进度

        参数:
            step: 增加的步数
            description: 当前步骤描述
        """
        self.current += step
        percent = self.current / self.total
        bar_length = 40
        filled = int(bar_length * percent)
        bar = '█' * filled + '░' * (bar_length - filled)

        # 计算耗时
        elapsed = time.time() - self.start_time
        eta = elapsed / percent * (1 - percent) if percent > 0 else 0

        # 构建输出
        output = f"\r{TerminalColors.OK_BLUE}[{self.prefix}] {TerminalColors.BOLD}{percent:>6.1%}{TerminalColors.ENDC} "
        output += f"{bar} {self.current}/{self.total} "

        if description:
            output += f"{description}"

        # 添加时间信息
        if eta > 0:
            output += f" ({elapsed:.1f}s ETA {eta:.1f}s)"
        else:
            output += f" ({elapsed:.1f}s)"

        sys.stdout.write(output)
        sys.stdout.flush()

    def finish(self, message: str = "完成!"):
        """完成进度条"""
        self.current = self.total
        percent = 1.0
        bar_length = 40
        bar = '█' * bar_length
        elapsed = time.time() - self.start_time

        output = f"\r{TerminalColors.OK_GREEN}[{self.prefix}] {TerminalColors.BOLD}{percent:>6.1%}{TerminalColors.ENDC} "
        output += f"{bar} {self.total}/{self.total} {message} ({elapsed:.2f}s)\n"

        sys.stdout.write(output)
        sys.stdout.flush()


def print_header():
    """打印程序标题"""
    header = f"""
╔══════════════════════════════════════════════════════════════════
║          Samdup 图像拼接算法 v2.0                          ║
║    视差感知的图割拼接 + SAM 智能分割                          ║
╚══════════════════════════════════════════════════════════════════
"""
    print(TerminalColors.HEADER + header + TerminalColors.ENDC)


def print_step(step_num: int, step_name: str, details: str = ""):
    """
    打印步骤信息

    参数:
        step_num: 步骤编号
        step_name: 步骤名称
        details: 详细信息
    """
    print(f"\n{TerminalColors.BOLD}[步骤 {step_num}/6] {step_name}{TerminalColors.ENDC}")
    if details:
        print(f"  {TerminalColors.OK_BLUE}➤{TerminalColors.ENDC} {details}")


def print_success(message: str, details: str = ""):
    """打印成功信息"""
    print(f"{TerminalColors.OK_GREEN}✓{TerminalColors.ENDC} {message}")
    if details:
        print(f"  {details}")


def print_warning(message: str, details: str = ""):
    """打印警告信息"""
    print(f"{TerminalColors.WARNING}⚠{TerminalColors.ENDC} {message}")
    if details:
        print(f"  {details}")


def print_info(message: str):
    """打印一般信息"""
    print(f"{TerminalColors.OK_BLUE}ℹ{TerminalColors.ENDC} {message}")


def print_stats(title: str, stats: dict):
    """
    打印统计信息

    参数:
        title: 标题
        stats: 统计信息字典
    """
    print(f"\n  {TerminalColors.BOLD}{title}:{TerminalColors.ENDC}")
    for key, value in stats.items():
        print(f"    • {key}: {value}")


# ============== 命令行参数解析 ==============

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    返回:
        args: 包含所有命令行参数的命名空间对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Samdup 图像拼接算法主程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python new_idea.py --img1 img1.jpg --img2 img2.jpg --output ./output
  python new_idea.py --img1 img1.jpg --img2 img2.jpg --scale 0.5 --ratio-test 0.98
        """
    )

    # 添加参数选项
    parser.add_argument(
        "--img1",
        default=DEFAULT_NEW_IDEA_IMG1,
        help="参考图像路径（基准图像）"
    )

    parser.add_argument(
        "--img2",
        default=DEFAULT_NEW_IDEA_IMG2,
        help="候选图像路径（待拼接图像）"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_NEW_IDEA_OUTPUT,
        help="输出目录路径"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"图像缩放比例（默认: {DEFAULT_SCALE}）"
    )

    parser.add_argument(
        "--ratio-test",
        type=float,
        default=DEFAULT_RATIO_TEST,
        help=f"SIFT 比例测试阈值（默认: {DEFAULT_RATIO_TEST}）"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式，减少输出信息"
    )

    return parser.parse_args()


def _infer_run_name(
    img_path: str,
    ratio_test: float,
    output_dir: str
) -> str:
    """
    推断本次运行的输出目录名称

    创建三层目录结构：图片名/阈值/递增数字
    例如：udis_000001/098/1

    参数:
        img_path: 输入图像路径
        ratio_test: 比例测试阈值
        output_dir: 输出根目录

    返回:
        run_name: 相对路径字符串，格式为 "图片名/阈值/数字"
    """
    # 将路径转换为 Path 对象
    img_path = Path(img_path)

    # ========== 获取图片名称 ==========
    # 检查路径结构：如果父目录是 rt_* 格式（阈值目录），则使用上上层目录名
    # 例如：path/to/udis_000001/rt_0.90/img.jpg -> 图片名为 udis_000001
    if img_path.parent.name.startswith("rt_") and img_path.parent.parent.name:
        img_name = img_path.parent.parent.name
    else:
        # 否则使用文件名（不含扩展名）
        img_name = img_path.stem

    # ========== 格式化阈值目录名 ==========
    # 将阈值格式化为两位小数，然后去除多余的零
    # 例如：0.98 -> "98", 0.75 -> "75"
    threshold_str = f"{ratio_test:.2f}".rstrip("0").rstrip(".")

    # ========== 确定递增数字 ==========
    # 构建基础路径：输出目录/图片名/阈值
    base_path = Path(output_dir) / img_name / threshold_str

    # 如果基础路径已存在，查找现有的数字目录
    if base_path.exists():
        # 收集所有纯数字命名的目录
        existing_numbers = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                existing_numbers.append(int(item.name))

        # 找到最大数字并加一，如果没有则从 1 开始
        next_num = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        # 基础路径不存在，从 1 开始
        next_num = 1

    # ========== 返回相对路径 ==========
    return f"{img_name}/{threshold_str}/{next_num}"


# ============== 主处理流程 ==============

def main() -> None:
    """
    主函数：执行完整的图像拼接流程

    流程步骤：
    1. 参数解析与初始化
    2. 图像加载与预处理
    3. SIFT 特征匹配
    4. RANSAC 单应性计算
    5. 视差区域检测
    6. 能量函数计算
    7. 图割拼接
    8. 结果保存
    """
    # ========== 步骤0：解析参数并打印欢迎信息 ==========
    args = parse_args()
    quiet_mode = args.quiet

    if not quiet_mode:
        print_header()

    # 记录开始时间
    total_start_time = time.time()

    # ========== 步骤1：初始化输出目录 ==========
    if not quiet_mode:
        print_info("初始化输出目录...")

    # 推断输出目录名称（三层结构）
    run_name = _infer_run_name(args.img1, args.ratio_test, args.output)

    # 创建输出目录
    output_dir = Path(ensure_dir(Path(args.output) / run_name))
    output_path = str(output_dir)

    if not quiet_mode:
        print_success("输出目录已创建", output_path)

    # 初始化进度条（11个主要步骤）
    progress = ProgressBar(11, "拼接进度")

    # ========== 步骤2：图像加载与预处理 ==========
    progress.update(1, "加载图像")
    # 读取参考图像（img1）和候选图像（img2）
    img1 = cv.imread(args.img1)
    img2 = cv.imread(args.img2)

    # 检查图像是否成功加载
    if img1 is None:
        raise FileNotFoundError(f"无法加载图像: {args.img1}")
    if img2 is None:
        raise FileNotFoundError(f"无法加载图像: {args.target}")

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 图像缩放处理
    scale = args.scale
    img1 = cv.resize(img1, (int(scale * w1), int(scale * h1)))
    img2 = cv.resize(img2, (int(scale * w2), int(scale * h2)))

    if not quiet_mode:
        print_success("图像加载完成", f"尺寸: {img1.shape}, 缩放: {scale}")

    # 获取比例测试阈值
    rt = args.ratio_test

    # ========== 步骤3：SIFT 初始特征匹配 ==========
    progress.update(1, "SIFT 特征匹配")
    if not quiet_mode:
        print_step(1, "SIFT 特征匹配", "检测关键点并计算匹配")

    src, dst, good, scores = get_initial_matches(img1, img2, rt)

    if not quiet_mode:
        print_success("特征匹配完成", f"检测到 {good} 对匹配点")

    # ========== 步骤4：提取视差区域点（提议方法） ==========
    progress.update(1, "提取视差区域点")
    if not quiet_mode:
        print_step(2, "提取视差区域点", "分析匹配点几何一致性")

    src_points, dst_points, H, src_after_real_H4, H4matches, left_parallax, right_parallax = get_true_points(
        img1, img2, src, dst, good, scores
    )

    if not quiet_mode:
        print_success("视差区域点提取完成", f"检测到 {len(left_parallax)} 个左视差点, {len(right_parallax)} 个右视差点")

    # ========== 步骤5：生成初步拼接结果（提议方法） ==========
    progress.update(1, "生成拼接结果")
    if not quiet_mode:
        print_step(3, "生成初步拼接结果", "应用单应性矩阵")

    stiching_2 = get_stitched_image(img2, img1, H)

    # ========== 步骤6：透视变换与裁剪 ==========
    progress.update(1, "透视变换")
    if not quiet_mode:
        print_step(4, "透视变换", "计算画布尺寸并变换图像")

    warp_img1, a, b, warp_all = get_warp(img2, img1, H)

    # 判断拼接方向
    direction = "左侧" if b == 0 else "右侧"
    if not quiet_mode:
        print_success("透视变换完成", f"候选图像在{direction}")

    # ========== 步骤7：SAM 视差区域检测 ==========
    progress.update(1, "检测视差区域")
    image = warp_img1

    # 检查是否存在视差区域点
    if len(left_parallax) != 0:
        # 使用 SAM 检测左侧视差区域
        left_parallax_shadow = find_parallax(image, left_parallax)
        # 使用 SAM 检测右侧视差区域
        right_parallax_shadow = find_parallax(img2, right_parallax)

        # 合并左右视差区域
        c = left_parallax_shadow + right_parallax_shadow

        # 转换为 uint8 类型
        c = (c * 255).astype(np.uint8)

        # 处理 RGBA 到 RGB 的转换
        if c.shape[-1] == 4:  # 如果是 4 通道（RGBA）
            c = c[:, :, :3]  # 只保留前 3 个通道（RGB）

        # 标记使用了视差掩码
        flag = True

        # 查找视差区域的边界
        new = find_area(c)

        if not quiet_mode:
            print_success("SAM 视差检测完成", "检测到视差区域")
    else:
        # 没有视差区域点，创建空掩码
        new = np.zeros(warp_img1.shape)
        flag = False

        if not quiet_mode:
            print_warning("未检测到视差区域", "使用常规拼接")

    # ========== 步骤8：RANSAC 单应性计算 ==========
    progress.update(1, "RANSAC 单应性计算")
    if not quiet_mode:
        print_step(5, "RANSAC 单应性计算", "过滤离群点")

    src_pts_left, src_after_real_RANSAC_H4, dst_RANSAC_points, matches1, M = get_RANSAC_points(
        img1, img2, src, dst, good
    )

    # 获取 RANSAC 内点数量
    ransac_inliers = int(np.sum(M is not None))
    if hasattr(all, 'cnt'):
        ransac_inliers = all.cnt

    # 生成 RANSAC 拼接结果
    stiching_ransac = get_stitched_image(img2, img1, M)

    if not quiet_mode:
        print_success("RANSAC 完成", f"保留 {ransac_inliers} 个内点")

    # ========== 步骤9：图像预处理（高斯模糊） ==========
    progress.update(1, "图像预处理")
    # 对参考图像应用高斯模糊以减少噪声
    gaussian_img2 = cv.GaussianBlur(
        img2,
        (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
        GAUSSIAN_SIGMA
    )

    # ========== 步骤10：查找重叠区域 ==========
    progress.update(1, "查找重叠区域")
    result_img1, result_img2, m, n, new = find_overlap_region(
        warp_img1, gaussian_img2, new, flag
    )

    overlap_width = result_img1.shape[1]
    if not quiet_mode:
        print_success("重叠区域检测完成", f"宽度: {overlap_width} 像素")

    # ========== 步骤11：能量函数计算 ==========
    progress.update(1, "计算能量函数")
    weight, weight1 = get_cost(
        result_img1, result_img2, dst_points, b, src_after_real_H4, new, flag
    )

    if not quiet_mode:
        print_success("能量函数计算完成", f"权重范围: [{weight.min():.1f}, {weight.max():.1f}]")

    # ========== 步骤12：图割拼接 ==========
    progress.update(1, "图割拼接")
    if not quiet_mode:
        print_step(6, "图割拼接", "寻找最优接缝")

    overlap, overlap1, seam_output, H4stitching = graph_cut(
        result_img1, result_img2, weight, weight1, a, b, m, n, img1, img2, H
    )

    # 完成进度条
    progress.finish("拼接完成")

    # ========== 计算总耗时 ==========
    total_time = time.time() - total_start_time

    # ========== 输出最终统计信息 ==========
    if not quiet_mode:
        print("\n" + "="*60)
        print(f"{TerminalColors.BOLD}拼接完成！{TerminalColors.ENDC}")
        print(f"  总耗时: {TerminalColors.OK_GREEN}{total_time:.2f} 秒{TerminalColors.ENDC}")
        print("="*60)

        # 打印关键统计
        stats = {
            "初始匹配点数": good,
            "最终匹配点数": all.real_index_number if hasattr(all, 'real_index_number') else "N/A",
            "RANSAC内点数": all.cnt if hasattr(all, 'cnt') else "N/A",
            "最终平均残差": f"{all.final_residuals:.4f}" if hasattr(all, 'final_residuals') else "N/A",
        }
        print_stats("统计信息", stats)

    # ========== 步骤13：保存结果图像 ==========
    if not quiet_mode:
        print_info("保存结果图像...")

    SaveImage(
        img1, img2, output_path, H4matches, warp_img1, overlap, overlap1, seam_output,
        H4stitching, matches1=matches1
    )

    if not quiet_mode:
        print_success("所有结果已保存", f"位置: {output_path}")


# ============== 程序入口点 ==============

if __name__ == '__main__':
    """
    程序入口点

    当脚本直接运行时，执行 main() 函数
    """
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{TerminalColors.WARNING}用户中断{TerminalColors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{TerminalColors.ERROR}程序出错: {e}{TerminalColors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
