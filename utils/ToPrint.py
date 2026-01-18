# -*- coding: utf-8 -*-
"""
统计信息输出模块
===========
功能说明：
1. 打印拼接管线的统计信息
2. 输出各种残差分析结果
3. 保存统计信息到文件

作者：gyd
日期：2023
"""

import sys  # 系统模块
from pathlib import Path  # 路径操作
from typing import Optional  # 类型注解

# 导入全局统计变量
import all

# 导入配置模块
from config import ensure_dir


# ============== 统计输出函数 ==============

def ToPrint(b: int, path: str) -> None:
    """
    打印并保存拼接管线的统计信息

    该函数会输出各个阶段的匹配点数量和残差信息

    参数:
        b: 候选图像位置标识
            - b == 0: candidate 图在左侧
            - b != 0: candidate 图在右侧
        path: 输出目录路径，统计信息将保存到 print.txt 文件中

    输出内容:
        - Proposed 算法的最终匹配点数量
        - 各阶段 (RT, A, H, H1, H2, H3) 的平均残差
        - Proposed 算法的最终平均残差
        - RANSAC 算法的匹配点数量
        - RANSAC 算法的平均残差
    """
    # 将路径转换为 Path 对象并确保目录存在
    path_obj = Path(path)
    ensure_dir(path_obj)

    # 设置输出文件路径
    print_file_path = path_obj / "print.txt"

    # 创建 Logger 实例用于双输出
    class Logger:
        """双输出日志记录器"""
        def __init__(self, file_path: str):
            """初始化日志记录器"""
            self.terminal = sys.stdout  # 保存终端引用
            self.log = open(file_path, "a", encoding='utf-8')  # 打开日志文件

        def write(self, message: str) -> None:
            """同时写入终端和文件"""
            self.terminal.write(message)
            self.log.write(message)

        def flush(self) -> None:
            """刷新缓冲区"""
            self.log.flush()

        def close(self) -> None:
            """关闭日志文件"""
            if hasattr(self, 'log'):
                self.log.close()

    # 创建并设置 Logger
    logger = Logger(str(print_file_path))
    sys.stdout = logger

    # ============== Proposed 算法输出 ==============

    # 打印我们算法的最终匹配点数量
    print("我们的算法最终一共有", all.real_index_number, "个点")
    print("                                                                                                  ")

    # 打印各个阶段的残差
    # RT 变换后的平均残差
    print("RT剔点后平均残差", all.RT_residuals)

    # A 变换后的平均残差
    print("A剃点后平均残差", all.A_residuals)

    # H 变换后的平均残差
    print("H剃点后平均残差", all.H_residuals)

    # H1 变换后的平均残差
    print("H1剃点后平均残差", all.H1_residuals)

    # H2 变换后的平均残差
    print("H2剃点后平均残差", all.H2_residuals)

    # H3 变换后的平均残差
    print("H3剃点后平均残差", all.H3_residuals)

    # 我们算法的最终平均残差
    print("我们的算法最终平均残差", all.final_residuals)
    print("                                                                                                  ")

    # ============== RANSAC 算法输出 ==============

    # RANSAC 的匹配点数量
    print("RANSAC最终得到", all.cnt, "个")

    # RANSAC 的平均残差
    print("RANSAC最终平均残差是", all.Ransac_residuals)
    print("                                                                                                  ")

    # ============== 候选图像位置判断 ==============

    # 判断 candidate 图在哪边
    # b 是偏移量，表示 candidate 图相对位置
    # if b == 0:
    #     print("candidate图在右侧")
    # else:
    #     print("candidate图在左侧")
    # print("                                                                                                  ")

    # 关闭日志文件并恢复标准输出
    logger.close()
    sys.stdout = logger.terminal  # 恢复原始的标准输出


def print_statistics() -> None:
    """
    打印当前所有统计信息的摘要

    该函数会向终端输出一个简洁的统计摘要，不写入文件
    """
    print("=" * 50)
    print("拼接管线统计摘要")
    print("=" * 50)

    # Proposed 算法统计
    print(f"Proposed 算法匹配点数: {all.real_index_number}")
    print(f"Proposed 算法平均残差: {all.final_residuals:.2f}")

    # 各阶段残差
    print(f"  - RT 阶段残差: {all.RT_residuals:.2f}")
    print(f"  - A 阶段残差: {all.A_residuals:.2f}")
    print(f"  - H 阶段残差: {all.H_residuals:.2f}")
    print(f"  - H1 阶段残差: {all.H1_residuals:.2f}")
    print(f"  - H2 阶段残差: {all.H2_residuals:.2f}")
    print(f"  - H3 阶段残差: {all.H3_residuals:.2f}")

    # RANSAC 算法统计
    print(f"RANSAC 匹配点数: {all.cnt}")
    print(f"RANSAC 平均残差: {all.Ransac_residuals:.2f}")

    print("=" * 50)


def get_statistics_dict() -> dict:
    """
    获取所有统计信息的字典形式

    返回:
        包含所有统计信息的字典，便于程序访问
    """
    return {
        'proposed': {
            'match_count': all.real_index_number,
            'final_residual': all.final_residuals,
            'rt_residual': all.RT_residuals,
            'a_residual': all.A_residuals,
            'h_residual': all.H_residuals,
            'h1_residual': all.H1_residuals,
            'h2_residual': all.H2_residuals,
            'h3_residual': all.H3_residuals,
        },
        'ransac': {
            'match_count': all.cnt,
            'residual': all.Ransac_residuals,
        },
    }


def reset_statistics() -> None:
    """
    重置所有全局统计变量

    在开始新的处理任务前调用此函数，确保统计数据的准确性
    """
    all.RT_residuals = 0
    all.A_residuals = 0
    all.H_residuals = 0
    all.H1_residuals = 0
    all.H2_residuals = 0
    all.H3_residuals = 0
    all.real_index_number = 0
    all.false_index_number = 0
    all.final_residuals = 0
    all.Ransac_residuals = 0
    all.cnt = 0
    all.true_points_counts = 0

    print("统计信息已重置")
