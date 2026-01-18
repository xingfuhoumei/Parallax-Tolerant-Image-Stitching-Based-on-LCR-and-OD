# -*- coding: utf-8 -*-
"""
日志记录模块
===========
功能说明：
1. 提供控制台和文件双重输出的日志功能
2. 自动创建日志目录
3. 支持同时输出到终端和日志文件

作者：gyd
日期：2023
"""

import sys  # 系统模块，用于重定向标准输出
from pathlib import Path  # 路径操作
from typing import Optional, TextIO  # 类型注解

# 从配置模块导入日志文件路径和目录创建函数
from config import DEFAULT_LOG_FILE, ensure_dir


class Logger:
    """
    双输出日志记录器类

    该类将输出同时写入终端和日志文件，确保所有程序输出都被记录
    """

    def __init__(self, filename: str = "Default.log") -> None:
        """
        初始化日志记录器

        参数:
            filename: 日志文件名称
        """
        # 保存标准输出（终端）的引用
        self.terminal: TextIO = sys.stdout
        # 以追加模式打开日志文件
        # 'a' 表示追加写入，不会覆盖已有内容
        self.log: TextIO = open(filename, "a", encoding='utf-8')

    def write(self, message: str) -> None:
        """
        写入消息到终端和日志文件

        参数:
            message: 要写入的消息字符串
        """
        # 将消息写入终端
        self.terminal.write(message)
        # 将消息写入日志文件
        self.log.write(message)

    def flush(self) -> None:
        """
        刷新缓冲区

        确保所有待写入的内容都被立即写入文件
        """
        # 刷新日志文件的缓冲区
        self.log.flush()
        # 刷新终端的缓冲区（如果有）
        if hasattr(self.terminal, 'flush'):
            self.terminal.flush()

    def close(self) -> None:
        """
        关闭日志文件

        在程序结束前调用，确保资源正确释放
        """
        if hasattr(self, 'log') and self.log is not None:
            self.log.close()

    def __enter__(self):
        """
        上下文管理器入口，支持 with 语句
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口，确保资源释放
        """
        self.close()


def setup_logging(log_file: Optional[str] = None) -> Logger:
    """
    设置日志系统并返回 Logger 实例

    该函数会：
    1. 确保日志目录存在
    2. 创建 Logger 实例
    3. 重定向标准输出到 Logger

    参数:
        log_file: 日志文件路径，如果为 None 则使用默认路径

    返回:
        Logger 实例
    """
    # 如果没有提供日志文件路径，使用默认配置
    if log_file is None:
        log_file = DEFAULT_LOG_FILE

    # 将路径转换为 Path 对象
    log_path = Path(log_file)

    # 确保日志文件的父目录存在
    ensure_dir(log_path.parent)

    # 创建 Logger 实例
    logger = Logger(str(log_path))

    # 重定向标准输出到 Logger
    sys.stdout = logger

    return logger


def restore_stdout() -> None:
    """
    恢复标准输出到终端

    在某些场景下，如果需要将输出恢复到终端，调用此函数
    """
    # 获取原始的标准输出（文件描述符 1）
    sys.stdout = sys.__stdout__


# ============== 模块初始化 ==============

# 当模块被导入时，自动设置日志系统
if __name__ != "__main__":
    # 获取日志文件路径
    log_path = Path(DEFAULT_LOG_FILE)

    # 确保日志目录存在
    ensure_dir(log_path.parent)

    # 创建 Logger 并重定向标准输出
    sys.stdout = Logger(str(log_path))
