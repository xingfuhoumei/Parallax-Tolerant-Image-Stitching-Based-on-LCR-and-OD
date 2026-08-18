# Parallax Tolerant Image Stitching Based on Concise Registration

本仓库用于视差鲁棒的图像拼接实验，包含特征匹配、视差区域估计、能量最小化与拼接等模块，并集成了 SAM 相关示例。
因为几年写了很多次版本，代码有点混乱，用ai重构了一下代码，有疑问等可以联系我的邮箱：1522623899@qq.com
也是纪念自己研究生的工作吧，没什么天赋却不愿意放弃，拼接方向确实复杂，不容易，后来的朋友加油。


## 更新日志

### 2025-01-19 (v2.0)
- 重构主程序 `new_idea.py`，添加动态进度条和彩色终端输出
- 修复 RANSAC 统计信息显示为 N/A 的问题
- 修复 `find_parallax` 函数在输入为空时的 TypeError
- 修复 `find_approxPolyDP.py` 中的 `cv` 未定义错误
- 统一 `all` 模块全局变量的正确设置
- 添加 `--quiet` 安静模式选项
- 优化异常处理和错误信息输出

## 目录结构
- `main.py`：SAM 单图分割演示（点/多点提示）
- `compare.py` / `new_idea.py`：拼接流程入口脚本
- `matches/`：特征匹配与 RANSAC 等
  - `Sift.py`：SIFT 特征提取与初始匹配
  - `Ransac.py`：RANSAC 单应性计算
  - `parallax_area.py`：视差区域点提取
- `parallax/`：视差区域与 SAM 相关工具
  - `SAM.py`：SAM 视差区域检测
  - `find_approxPolyDP.py`：视差区域边界查找
- `stitch/`：能量函数、GraphCut 及拼接
  - `energy_fuction.py`：能量函数计算
  - `graphcut.py`：图割拼接
  - `stitching_function.py`：透视变换与裁剪
- `utils/`：日志与工具函数
  - `ToPrint.py`：统计信息输出
  - `save_img.py`：图像保存

## 依赖
```bash
pip install -r requirements.txt
```

## 数据与模型
- 默认输入/输出路径在 `config.py` 中定义，可通过环境变量覆盖：
  - `SAMDUP_DATA_ROOT`，`SAMDUP_OUTPUT_DIR`，`SAMDUP_LOG_DIR`
  - `SAMDUP_SAM_CHECKPOINT`（SAM 权重路径）
- 需要准备 SAM 权重文件（例如 `sam_vit_b_01ec64.pth`）并设置 `SAMDUP_SAM_CHECKPOINT` 或放到项目根目录

## 使用示例

### SAM 单图分割演示
```bash
python main.py --image /path/to/image.png --sam-checkpoint /path/to/sam_vit_b_01ec64.pth
```

### 图像拼接
```bash
# 基本用法
python new_idea.py

# 指定输入图像和输出目录
python new_idea.py --img1 /path/to/img1.jpg --img2 /path/to/img2.jpg --output ./output

# 完整参数示例
SAMDUP_SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth python new_idea.py \
    --img1 /data/gyd/paper/Samdup4/outputs1/udis/udis_000001/rt_0.75/1-candidate.jpg \
    --img2 /data/gyd/paper/Samdup4/outputs1/udis/udis_000001/rt_0.75/2-reference.jpg \
    --output /data/gyd/paper/Samdup2/outputs \
    --scale 1 \
    --ratio-test 0.7

# 安静模式（最小化输出）
python new_idea.py --img1 img1.jpg --img2 img2.jpg --output ./output --quiet
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--img1` | 候选图像路径（基准图像） | `config.py` 中定义 |
| `--img2` | 参考图像路径（待拼接图像） | `config.py` 中定义 |
| `--output` | 输出目录路径 | `./outputs` |
| `--scale` | 图像缩放比例 | `0.5` |
| `--ratio-test` | SIFT 比例测试阈值 | `0.98` |
| `--quiet` | 安静模式，减少输出信息 | `False` |

### 输出文件说明
输出目录包含以下文件：
- `1-candidate.jpg` - 候选图像
- `2-reference.jpg` - 参考图像
- `3-ransac_mp.jpg` - RANSAC 匹配点可视化
- `4-proposed_mp.jpg` - 提议方法匹配点可视化
- `5-overlap_energy.jpg` - 重叠区域能量图
- `6-seam_output.jpg` - 接缝输出可视化
- `7-overlap.jpg` - 重叠区域结果
- `8-H4stitching.jpg` - 最终拼接结果
- `print.txt` - 统计信息（匹配点数、残差等）

## 备注
部分脚本依赖 CUDA/显卡环境（如 `torch` + `segment-anything`），请按需配置。
