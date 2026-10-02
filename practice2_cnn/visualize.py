"""可视化卷积层输出的部分特征图（浅层 vs 深层）。

用法（工作目录 = practice2_cnn）：
    # 从测试集随机抽 3 张
    python visualize.py --ckpt outputs/best_full_aug.pt --split test --n 3

    # 指定图片
    python visualize.py --ckpt outputs/best_full_aug.pt --img data_split/test/cat/cat.10004.jpg

产物：outputs/feature_maps.png
    每张图占 5 行 x 8 列：第 1 行是输入，后 4 行依次是 block1~block4 的前 8 个通道。
    行标签写在第一列的 y 轴上（注意：不能用 ax.axis("off")，否则 ylabel 会被一起关掉）。
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import eval_transform
from model import SimpleCNN


def denormalize(t, mean, std):
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(std) + np.array(mean)
    return np.clip(x, 0, 1)


def collect_images(args):
    """返回 [(路径, 类别名或 None), ...]"""
    items = []
    if args.img:
        for p in args.img:
            items.append((Path(p), None))
        return items

    root = Path(args.data_root) / args.split
    files = sorted(root.glob("*/*.jpg")) + sorted(root.glob("*/*.png"))
    random.seed(args.seed)
    picked = random.sample(files, min(args.n, len(files)))
    for p in picked:
        items.append((p, p.parent.name))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img", type=str, nargs="*", default=None,
                        help="一张或多张图片；不填则从 --split 里随机抽 --n 张")
    parser.add_argument("--data_root", type=str, default="data_split")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cols", type=int, default=8, help="每行展示多少个通道")
    parser.add_argument("--out", type=str, default="outputs/feature_maps.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    from dataset import MEAN, STD
    items = collect_images(args)
    block_names = ["block1", "block2", "block3", "block4"]
    rows_per_img = 1 + len(block_names)
    cols = args.cols

    fig, axes = plt.subplots(rows_per_img * len(items), cols,
                             figsize=(1.55 * cols, 1.55 * rows_per_img * len(items)))
    if len(items) == 1:
        axes = axes.reshape(rows_per_img, cols)

    for k, (path, label) in enumerate(items):
        img = Image.open(path).convert("RGB")
        x = eval_transform(img).unsqueeze(0).to(device)

        acts = []
        with torch.no_grad():
            h = x
            for layer in model.features:
                h = layer(h)
                acts.append(h)

        r0 = k * rows_per_img
        # --- 输入行 ---
        axes[r0, 0].imshow(denormalize(x[0], MEAN, STD))
        axes[r0, 0].set_ylabel("input\n3x128x128", fontsize=8)
        tag = path.name + (f"\nGT: {label}" if label else "")
        axes[r0, cols // 2].set_title(tag, fontsize=10)
        for j in range(cols):
            axes[r0, j].set_xticks([]); axes[r0, j].set_yticks([])
            for s in axes[r0, j].spines.values():
                s.set_visible(False)

        # --- 4 个 block 的特征图 ---
        for i, (a, name) in enumerate(zip(acts, block_names)):
            a = a[0]                      # C,H,W
            row = r0 + 1 + i
            for j in range(cols):
                ax = axes[row, j]
                if j < a.shape[0]:
                    ax.imshow(a[j].cpu(), cmap="viridis")   # 每个通道单独自动缩放
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            axes[row, 0].set_ylabel(f"{name}\n{a.shape[0]}x{a.shape[1]}x{a.shape[2]}",
                                    fontsize=8)

    fig.suptitle("Feature maps: input + first 8 channels of each block "
                 "(each channel auto-scaled)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"共 {len(items)} 张图，保存到 {args.out}")


if __name__ == "__main__":
    main()
