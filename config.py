# -*- coding: utf-8 -*-
"""
配置管理模块
===========
功能说明：
1. 集中管理所有配置参数
2. 支持环境变量覆盖默认配置
3. 提供路径管理和目录创建功能

作者：gyd
日期：2023
"""

import os  # 操作系统接口，用于读取环境变量
from pathlib import Path  # 面向对象的路径操作库
from typing import Union, Optional  # 类型注解支持


# ============== 项目根目录配置 ==============

# 获取当前文件所在目录的父目录作为项目根目录
# __file__ 表示当前文件的路径
# resolve() 方法将路径转换为绝对路径，并解析所有符号链接
# .parent 获取父目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ============== 类型定义 ==============

# 定义 PathOrStr 类型，表示可以是 Path 对象或字符串
PathOrStr = Union[Path, str]


# ============== 辅助函数 ==============

def _env_or(name: str, default: Union[str, Path]) -> Union[str, Path]:
    """
    从环境变量获取配置，如果不存在则返回默认值

    参数:
        name: 环境变量名称
        default: 默认值

    返回:
        环境变量的值或默认值
    """
    # 获取环境变量，如果不存在或为空字符串则返回默认值
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_path(name: str, default: Union[str, Path]) -> Path:
    """
    从环境变量获取路径配置，返回 Path 对象

    参数:
        name: 环境变量名称
        default: 默认路径

    返回:
        Path 对象
    """
    # 获取环境变量值并转换为 Path 对象
    return Path(_env_or(name, default))


# ============== 核心路径配置 ==============

# 数据根目录：存放输入图像和数据的目录
# 可通过环境变量 SAMDUP_DATA_ROOT 覆盖
DATA_ROOT: Path = _env_path("SAMDUP_DATA_ROOT", PROJECT_ROOT / "data")

# 输出目录：存放处理结果的目录
# 可通过环境变量 SAMDUP_OUTPUT_DIR 覆盖
OUTPUT_DIR: Path = _env_path("SAMDUP_OUTPUT_DIR", PROJECT_ROOT / "outputs")

# 日志目录：存放日志文件的目录
# 可通过环境变量 SAMDUP_LOG_DIR 覆盖
LOG_DIR: Path = _env_path("SAMDUP_LOG_DIR", PROJECT_ROOT / "logs")

# SAM（Segment Anything Model）模型检查点路径
# 可通过环境变量 SAMDUP_SAM_CHECKPOINT 覆盖
SAM_CHECKPOINT: Path = _env_path("SAMDUP_SAM_CHECKPOINT", PROJECT_ROOT / "sam_vit_b_01ec64.pth")

# 调试输出目录：存放调试中间结果的目录
# 可通过环境变量 SAMDUP_DEBUG_DIR 覆盖
DEFAULT_DEBUG_DIR: Path = _env_path("SAMDUP_DEBUG_DIR", PROJECT_ROOT / "debug")


# ============== 主程序默认配置 ==============

# 主程序输入图像路径（用于 SAM 单图像分割演示）
DEFAULT_MAIN_IMAGE: str = _env_or(
    "SAMDUP_MAIN_IMAGE",
    str(DATA_ROOT / "main_image.png"),
)

# 对比实验的第一张图像路径
DEFAULT_COMPARE_IMG1: str = _env_or(
    "SAMDUP_COMPARE_IMG1",
    str(DATA_ROOT / "compare_img1.jpg"),
)

# 对比实验的第二张图像路径
DEFAULT_COMPARE_IMG2: str = _env_or(
    "SAMDUP_COMPARE_IMG2",
    str(DATA_ROOT / "compare_img2.jpg"),
)

# 对比实验输出目录
DEFAULT_COMPARE_OUTPUT: str = _env_or(
    "SAMDUP_COMPARE_OUTPUT",
    str(OUTPUT_DIR / "compare"),
)

# new_idea 管线的第一张图像路径（参考图像）
DEFAULT_NEW_IDEA_IMG1: str = _env_or(
    "SAMDUP_NEW_IDEA_IMG1",
    str(DATA_ROOT / "new_idea_img1.jpg"),
)

# new_idea 管线的第二张图像路径（候选图像）
DEFAULT_NEW_IDEA_IMG2: str = _env_or(
    "SAMDUP_NEW_IDEA_IMG2",
    str(DATA_ROOT / "new_idea_img2.jpg"),
)

# new_idea 管线的输出目录
DEFAULT_NEW_IDEA_OUTPUT: str = _env_or(
    "SAMDUP_NEW_IDEA_OUTPUT",
    str(OUTPUT_DIR),
)

# 测试程序的第一张图像路径
DEFAULT_TEST_IMG1: str = _env_or(
    "SAMDUP_TEST_IMG1",
    str(DATA_ROOT / "test_img1.png"),
)

# 测试程序的第二张图像路径
DEFAULT_TEST_IMG2: str = _env_or(
    "SAMDUP_TEST_IMG2",
    str(DATA_ROOT / "test_img2.png"),
)

# 日志文件路径
DEFAULT_LOG_FILE: str = _env_or(
    "SAMDUP_LOG_FILE",
    str(LOG_DIR / "run.log"),
)

# 调试目录路径
DEFAULT_DEBUG_DIR: str = _env_or(
    "SAMDUP_DEBUG_DIR",
    str(OUTPUT_DIR / "debug"),
)


# ============== 工具函数 ==============

def ensure_dir(path: Union[str, Path]) -> str:
    """
    确保目录存在，如果不存在则创建

    该函数会递归创建所有不存在的父目录

    参数:
        path: 目录路径（可以是字符串或Path对象）

    返回:
        目录路径的字符串表示

    示例:
        >>> ensure_dir("/tmp/test/subdir")
        '/tmp/test/subdir'
    """
    # 将输入转换为 Path 对象
    path_obj = Path(path)
    # exist_ok=True 表示如果目录已存在也不报错
    # parents=True 表示递归创建所有父目录
    path_obj.mkdir(parents=True, exist_ok=True)
    # 返回字符串形式的路径
    return str(path_obj)


def get_config() -> dict:
    """
    获取所有配置的字典形式

    返回:
        包含所有配置项的字典，便于程序访问配置

    示例:
        >>> config = get_config()
        >>> print(config['DATA_ROOT'])
        PosixPath('/path/to/data')
    """
    return {
        'PROJECT_ROOT': PROJECT_ROOT,
        'DATA_ROOT': DATA_ROOT,
        'OUTPUT_DIR': OUTPUT_DIR,
        'LOG_DIR': LOG_DIR,
        'SAM_CHECKPOINT': SAM_CHECKPOINT,
        'DEFAULT_MAIN_IMAGE': DEFAULT_MAIN_IMAGE,
        'DEFAULT_COMPARE_IMG1': DEFAULT_COMPARE_IMG1,
        'DEFAULT_COMPARE_IMG2': DEFAULT_COMPARE_IMG2,
        'DEFAULT_COMPARE_OUTPUT': DEFAULT_COMPARE_OUTPUT,
        'DEFAULT_NEW_IDEA_IMG1': DEFAULT_NEW_IDEA_IMG1,
        'DEFAULT_NEW_IDEA_IMG2': DEFAULT_NEW_IDEA_IMG2,
        'DEFAULT_NEW_IDEA_OUTPUT': DEFAULT_NEW_IDEA_OUTPUT,
        'DEFAULT_TEST_IMG1': DEFAULT_TEST_IMG1,
        'DEFAULT_TEST_IMG2': DEFAULT_TEST_IMG2,
        'DEFAULT_LOG_FILE': DEFAULT_LOG_FILE,
        'DEFAULT_DEBUG_DIR': DEFAULT_DEBUG_DIR,
    }


def validate_config() -> bool:
    """
    验证关键配置是否有效

    检查：
    1. SAM 模型文件是否存在
    2. 输出目录是否可写

    返回:
        配置是否有效
    """
    # 检查 SAM 模型文件
    if not SAM_CHECKPOINT.exists():
        print(f"警告: SAM 模型文件不存在: {SAM_CHECKPOINT}")
        return False

    # 检查输出目录是否可写
    try:
        ensure_dir(OUTPUT_DIR)
        # 尝试创建测试文件
        test_file = OUTPUT_DIR / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        print(f"警告: 输出目录不可写: {OUTPUT_DIR}, 错误: {e}")
        return False

    return True
