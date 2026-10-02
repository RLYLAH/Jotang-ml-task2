# Jotang-ml-task2

深度学习作业二：先用手写卷积拆开一张图片，再用 PyTorch 搭建 CNN 完成 Kaggle Dogs vs. Cats 猫狗二分类。

- 完整实验记录与问题解答：**[report.md](report.md)**
- 实践一代码：`practice1_conv/`
- 实践二代码：`practice2_cnn/`
- 所有结果图：`practice1_conv/outputs/`、`practice2_cnn/outputs/`

## 1. 环境

| 项目 | 值                                           |
|---|---------------------------------------------|
| 系统 | Windows 11 (10.0.26100)                     |
| Python | 3.13.11（conda 环境 ）                          |
| 解释器路径 | -                                           |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（CUDA 可用） |

依赖见 `practice2_cnn/requirements.txt`（只有 5 个包：torch / torchvision / numpy / pillow / matplotlib）。

> 注意：或许终端里的默认 `python` 是 conda base，**没有装 torch**。
> 运行前请先 `conda activate "某环境"`，或直接用上面的解释器全路径。

## 2. 数据

```
practice2_cnn/dc/train/   25000 张有标签图片（cat 12500 / dog 12500）
practice2_cnn/dc/test/    12500 张无标签图片（原测试集，无标签，故仅用于推理演示）
```

`dc/train` 由 `split_data.py` 按 **70/15/15 分层抽样** 拆成 `data_split/{train,val,test}/{cat,dog}`：

| split | cat | dog | 合计 |
|---|---|---|---|
| train | 8750 | 8750 | 17500 |
| val | 1875 | 1875 | 3750 |
| test | 1875 | 1875 | 3750 |

> `data_split/` 与 `data_split_sub2000/` 都是脚本生成的（默认用硬链接，不额外占空间），
> 可以随时删掉后重新生成；

## 3. 复现步骤

### 实践一：卷积核（无需数据集）

```powershell
cd practice1_conv
python read_image.py     # 打印图片的 Pillow/NumPy/Tensor 信息
python run_filters.py    # 4 种卷积核 + padding/stride 实验，出图到 outputs/
```

脚本内部按 `__file__` 解析路径，所以从任何工作目录运行都可以。

### 实践二：猫狗分类

```powershell
cd practice2_cnn
conda activate myenv(此仅为我自我的一个虚拟环境)

# 1) 划分数据集（默认全量；加 --subset 2000 可只取每类 2000 张做快速对照实验）
python split_data.py
python split_data.py --subset 2000 --dst data_split_sub2000

# 2) 观察数据 / 预处理 / 增强（出 3 张图，并打印尺寸与画质统计）
python preview_data.py

# 3) 训练主模型（全量 17500 张，20 epoch，带数据增强）
python train.py --data_root data_split --epochs 20 --tag full_aug --seed 42

# 4) 对照实验：子集上"有增强 vs 无增强"（唯一变量是增强，种子固定）
python train.py --data_root data_split_sub2000 --epochs 15 --tag subset_aug   --augment 1 --seed 42
python train.py --data_root data_split_sub2000 --epochs 15 --tag subset_noaug --augment 0 --seed 42

# 5) 全连接层基线（趁热打铁第 1 题）
python fc_baseline.py --data_root data_split_sub2000 --epochs 15 --seed 42

# 6) 改进实验：强化增强，其余设置与主实验一致
python train.py --data_root data_split --epochs 20 --tag full_aug_v2 --strong_aug 1 --seed 42

# 7) 结果分析
python model_arch.py                                                          # 结构图 + 逐层形状表
python visualize.py --ckpt outputs/best_full_aug.pt --split test --n 3        # 特征图
python analyze_errors.py --ckpt outputs/best_full_aug.pt --split test --n 16  # 混淆矩阵 + 错分样本
python predict.py --ckpt outputs/best_full_aug.pt --img dc/test/1000.jpg      # 独立推理
```

## 4. 脚本与产物对照

| 脚本 | 作用 | 主要产物 |
|---|---|---|
| `practice1_conv/read_image.py` | 图片 → NumPy/Tensor | 控制台输出 |
| `practice1_conv/conv2d.py` | 手写二维卷积（只用 NumPy） | 被 `run_filters.py` 调用 |
| `practice1_conv/run_filters.py` | 均值/高斯/锐化/Sobel + padding/stride 实验 | `comparison.png` 等 9 张图 |
| `practice2_cnn/split_data.py` | 70/15/15 分层划分 | `data_split/` |
| `practice2_cnn/preview_data.py` | 数据观察、预处理与增强对照 | `data_overview.png`、`preprocess_preview.png`、`augment_preview.png` |
| `practice2_cnn/dataset.py` | 预处理流水线 + DataLoader | — |
| `practice2_cnn/model.py` | SimpleCNN 定义 | — |
| `practice2_cnn/model_arch.py` | 模型结构图与逐层形状表 | `model_arch.png` |
| `practice2_cnn/train.py` | 训练 / 验证 / 测试 | `best_*.pt`、`curves_*.png`、`log_*.txt` |
| `practice2_cnn/fc_baseline.py` | 全连接层基线 | `curves_fc_baseline.png`、`log_fc_baseline.txt` |
| `practice2_cnn/visualize.py` | 特征图可视化 | `feature_maps.png` |
| `practice2_cnn/analyze_errors.py` | 混淆矩阵 + 错分样本 | `confusion_matrix.png`、`errors.png` |
| `practice2_cnn/predict.py` | 独立推理 | `predict.png`、`predict_processed.png` |

## 5. 已知的一些约定或许

- 卷积实现是**互相关**（不翻转核），与 `nn.Conv2d` 一致；`padding="same"` 只在奇数核 + `stride=1` 时严格保持尺寸。
- 预处理在训练和推理时**完全一致**：`Resize(144)（短边缩放，不拉伸） → CenterCrop(128) → ToTensor → Normalize`（ImageNet 均值方差）。
  初版曾用 `Resize((144,144))`（元组）把图强行拉成正方形，会产生拉伸；`dataset.py` 里保留了 `eval_transform_stretch` 这一份历史实现，配合 `analyze_errors.py --transform stretch` 可以复现两版对比（见报告 §2.4）。
- 验证/测试**不做任何随机增强**，只做确定性的缩放和中心裁剪。
- 训练集、验证集、测试集严格分开；测试集只在最后评估一次，不用来调参。
- 所有对照实验都固定 `seed=42`；只有最早的 `full_aug` 主实验是在加入 `--seed` 之前跑的（日志 `log_full_aug.txt`）。
- 注意 `Resize(144)` 与 `Resize((144,144))` 只差一对括号但含义完全不同——前者保持长宽比，后者会变形。
