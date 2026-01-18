# Samdup2

本仓库用于视差鲁棒的图像拼接实验，包含特征匹配、视差区域估计、能量最小化与拼接等模块，并集成了 SAM 相关示例。

## 目录结构
- `main.py`：SAM 单图分割演示（点/多点提示）。
- `compare.py` / `new_idea.py`：拼接流程入口脚本。
- `matches/`：特征匹配与 RANSAC 等。
- `parallax/`：视差区域与 SAM 相关工具。
- `stitch/`：能量函数、GraphCut 及拼接。
- `utils/`：日志与工具函数。

## 依赖
```bash
pip install -r requirements.txt
```

## 数据与模型
- 默认输入/输出路径在 `config.py` 中定义，可通过环境变量覆盖：
  - `SAMDUP_DATA_ROOT`，`SAMDUP_OUTPUT_DIR`，`SAMDUP_LOG_DIR`
  - `SAMDUP_SAM_CHECKPOINT`（SAM 权重路径）
- 需要准备 SAM 权重文件（例如 `sam_vit_b_01ec64.pth`）并设置 `SAMDUP_SAM_CHECKPOINT` 或放到项目根目录。

## 使用示例
```bash
# SAM 单图分割演示
python main.py --image /path/to/image.png --sam-checkpoint /path/to/sam_vit_b_01ec64.pth

# 拼接流程（示例）
python compare.py
python new_idea.py
```

## 备注
部分脚本依赖 CUDA/显卡环境（如 `torch` + `segment-anything`），请按需配置。
