# 跨模态互补的脑肿瘤分割教学项目

这是一个面向 **0 基础本科生** 的教学型项目。项目目标不是追求最先进结果，而是帮助你完成一个“能理解、能运行、能写进论文”的脑肿瘤分割闭环。

## 1. 项目内容概览

本项目采用以下简化路线：

- 输入：4 个 MRI 序列（T1、T1ce、T2、FLAIR）。
- 任务：脑肿瘤整体区域 WT 二分类分割。
- 基线模型：4 通道输入的 2D U-Net。
- 改进模型：在输入端加入一个轻量 `ModalityGate` 模块，让模型自动学习“更相信哪种模态”。
- 真实数据：BraTS 官方数据集。
- 教学演示：自带 `generate_toy_dataset.py`，可先生成合成小数据跑通流程。

## 2. 目录说明

```text
brain_tumor_segmentation_project/
├── README.md                    # 中文使用说明
├── requirements.txt             # Python 依赖
├── check_env.py                 # 检查环境
├── generate_toy_dataset.py      # 生成合成 toy 数据
├── prepare_brats_slices.py      # 把 BraTS 体数据转成 2D 切片
├── train.py                     # 训练脚本
├── predict_case.py              # 单病例预测脚本
├── dataset.py                   # 数据集读取逻辑
├── losses.py                    # Dice+BCE 损失与指标
├── utils.py                     # 工具函数
├── models/
│   └── unet.py                  # baseline U-Net 与 gate 版本
└── docs/
    └── project_report.md        # 项目报告
```

## 3. 环境安装

### 3.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows 用户可以改成：

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

### 3.3 检查环境

```bash
python check_env.py
```

## 4. 第一次运行：先用 toy 数据集练手

### 4.1 生成 toy 数据

```bash
python generate_toy_dataset.py --output_dir toy_dataset
```

通俗解释：这一步会生成一批“假 MRI 数据”，目的是让你先学会整个项目怎么跑，不必一开始就下载真实医学数据。

### 4.2 训练 baseline 模型

```bash
python train.py --data_dir toy_dataset --model baseline --epochs 5 --batch_size 8
```

### 4.3 训练 gate 模型

```bash
python train.py --data_dir toy_dataset --model gate --epochs 5 --batch_size 8
```

通俗解释：

- `baseline` 就是普通 U-Net，4 个模态直接拼接输入。
- `gate` 会先给不同模态分配权重，相当于让网络学习“这一块更该相信哪一个 MRI 序列”。

## 5. 真实 BraTS 数据的使用流程

### 5.1 下载 BraTS 数据

你需要从官方渠道下载 BraTS 数据集。下载完成后，每个病例目录里一般包含：

```text
BraTS2021_00000/
├── BraTS2021_00000_t1.nii.gz
├── BraTS2021_00000_t1ce.nii.gz
├── BraTS2021_00000_t2.nii.gz
├── BraTS2021_00000_flair.nii.gz
└── BraTS2021_00000_seg.nii.gz
```

### 5.2 预处理成可训练的 2D 切片

```bash
python prepare_brats_slices.py --brats_root /path/to/BraTS2021 --output_dir data_npz --max_cases 20
```

这一步做的事情：

1. 读取 4 个 MRI 模态；
2. 对每个模态做标准化；
3. 把标签转成 WT 二分类；
4. 按病例划分 train / val / test；
5. 保存为 `.npz` 小文件，方便 2D 训练。

### 5.3 训练真实数据模型

```bash
python train.py --data_dir data_npz --model gate --epochs 20 --batch_size 8
```

## 6. 单病例预测

```bash
python predict_case.py --checkpoint outputs/gate_xxx/best_model.pt --case_dir /path/to/BraTS2021_00000 --output_dir predictions
```

输出内容：

- `*_pred_wt.nii.gz`：预测掩膜；
- `*_preview.png`：中间切片的可视化预览；
- `*_result.json`：预测记录。

## 7. 每个文件是做什么的

- `generate_toy_dataset.py`：生成练手数据。
- `prepare_brats_slices.py`：整理真实 BraTS 数据。
- `train.py`：训练模型并保存最佳权重。
- `predict_case.py`：加载训练好的模型并预测单个病例。
- `models/unet.py`：U-Net 主体 + 模态门控模块。
- `losses.py`：损失函数和 Dice/IoU 指标。

## 8. 建议学习顺序

1. 先安装环境；
2. 先跑 toy 数据；
3. 看懂输出目录中的 `history.json`、`test_metrics.json`、`training_curves.png`；
4. 再下载真实 BraTS 数据；
5. 用 20 个病例先实验；
6. 最后再写论文与项目报告。

## 9. 论文里可以怎么描述这个项目

你可以把它写成：

- 基线：4 通道输入的 2D U-Net；
- 改进：加入轻量跨模态权重分配模块，实现特征互补；
- 任务：多序列 MRI 的脑肿瘤 WT 分割；
- 指标：Dice、IoU；
- 实验：baseline vs gate，对比 + 消融 + 可视化。
