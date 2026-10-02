"""数据观察与预处理/增强对照（对应"实践二"的前两步）。

用法（工作目录 = practice2_cnn）：
    python preview_data.py                 # 全量尺寸统计 + 1000 张画质抽样
    python preview_data.py --quality_n 500

产物：
    outputs/data_overview.png     12 张随机原图，标题写着各自的 (宽 x 高)
    outputs/preprocess_preview.png 原图 / 验证集流水线 / 训练集流水线 三者对比
    outputs/augment_preview.png   默认增强、强化增强、过度增强各 4 次采样

同时打印（报告里的数字直接来自这里）：
    - 猫狗各多少张
    - 宽/高/宽高比的分布（min / 25% / 50% / 75% / max）
    - 颜色模式统计
    - 亮度、对比度、清晰度（拉普拉斯方差）的分布 -> 回答"清晰度、背景是否相同"
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
from torchvision import transforms

from dataset import MEAN, STD, train_transform, strong_train_transform, eval_transform

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dc" / "train"
OUT_DIR = BASE_DIR / "outputs"


def setup_cjk_font():
    """matplotlib 默认字体没有中文，这里挑一个系统里可用的中文字体。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun",
                 "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 统计
def scan_sizes(files):
    """只读文件头拿 (宽, 高, 模式)，不解码像素，25000 张也很快。"""
    sizes, modes = [], {}
    for p in files:
        with Image.open(p) as im:
            sizes.append((im.size[0], im.size[1]))
            modes[im.mode] = modes.get(im.mode, 0) + 1
    return np.array(sizes), modes


def describe(name, arr):
    q = np.percentile(arr, [0, 25, 50, 75, 100])
    print(f"   {name:<10} min={q[0]:8.1f}  25%={q[1]:8.1f}  中位数={q[2]:8.1f}  "
          f"75%={q[3]:8.1f}  max={q[4]:8.1f}  均值={arr.mean():8.1f}")


def quality_metrics(paths):
    """亮度 / 对比度 / 清晰度（拉普拉斯方差）。"""
    bright, contrast, sharp = [], [], []
    for p in paths:
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32)
        bright.append(g.mean())
        contrast.append(g.std())
        # 拉普拉斯算子（不用任何现成卷积库，直接切片差分）
        lap = (4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
               - g[1:-1, :-2] - g[1:-1, 2:])
        sharp.append(lap.var())
    return np.array(bright), np.array(contrast), np.array(sharp)


# ------------------------------------------------------------------ 画图
def denorm(t):
    x = t.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return np.clip(x, 0, 1)


def fig_overview(files, path):
    random.seed(0)
    cats = [f for f in files if f.name.startswith("cat.")]
    dogs = [f for f in files if f.name.startswith("dog.")]
    picked = random.sample(cats, 6) + random.sample(dogs, 6)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, p in zip(axes.ravel(), picked):
        with Image.open(p) as im:
            w, h = im.size
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{p.name}\n{w} x {h}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Random samples from dc/train (note the very different sizes/backgrounds)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=120)
    plt.close(fig)


def fig_preprocess(src_path, path):
    """同一张图：原图 / 验证集流水线 / 训练集流水线（反归一化后显示）。"""
    raw = Image.open(src_path).convert("RGB")
    shown = [("Original\n{}x{}".format(*raw.size), np.asarray(raw) / 255.0),
             ("eval_transform\nResize144+CenterCrop128", denorm(eval_transform(raw))),
             ("train_transform\nRandomResizedCrop+Flip+Jitter", denorm(train_transform(raw)))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (title, data) in zip(axes, shown):
        ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Preprocessing preview: {Path(src_path).name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def fig_augment(src_path, path):
    raw = Image.open(src_path).convert("RGB")
    over = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(128, scale=(0.12, 0.25)),   # 过度裁剪
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    rows = [("默认增强 (scale 0.7~1.0)", train_transform),
            ("强化增强 (scale 0.5~1.0 + rotation10 + erasing)", strong_train_transform),
            ("过度增强 (scale 0.12~0.25 + rotation45) —— 主体被切掉/转歪", over)]

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for r, (title, tf) in enumerate(rows):
        for c in range(4):
            axes[r, c].imshow(denorm(tf(raw)))
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=9, loc="left")
    fig.suptitle(f"Augmentation samples: {Path(src_path).name}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=str(TRAIN_DIR))
    parser.add_argument("--quality_n", type=int, default=1000,
                        help="抽样多少张计算亮度/对比度/清晰度（解码较慢，默认 1000）")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    setup_cjk_font()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_dir)
    files = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in
                    (".jpg", ".jpeg", ".png")])
    cats = [f for f in files if f.name.startswith("cat.")]
    dogs = [f for f in files if f.name.startswith("dog.")]
    print("=" * 70)
    print("1. 数据集规模")
    print(f"   dc/train 共 {len(files)} 张：cat {len(cats)} 张，dog {len(dogs)} 张")
    test_files = sorted([p for p in (train_dir.parent / "test").iterdir() if p.is_file()])
    print(f"   dc/test  共 {len(test_files)} 张（无标签，仅用于推理演示）")

    print("\n2. 图片尺寸分布（全量 %d 张，只读文件头）" % len(files))
    sizes, modes = scan_sizes(files)
    w, h = sizes[:, 0], sizes[:, 1]
    describe("宽 W", w.astype(float))
    describe("高 H", h.astype(float))
    describe("宽高比 W/H", (w / h).astype(float))
    describe("像素数 W*H", (w * h).astype(float))
    print(f"   颜色模式统计: {modes}")
    print(f"   宽高完全相同的图: {int(np.sum(w == h))} 张（占 {np.mean(w == h) * 100:.1f}%）")
    print(f"   最大的 3 张: {sorted(set(map(tuple, sizes)), reverse=True)[:3]}")
    print("   -> 尺寸差异很大，且绝大多数不是正方形，所以必须统一 resize；"
          "直接拉伸会改变物体长宽比，因此用 Resize+CenterCrop（评估）"
          "和 RandomResizedCrop（训练）。")

    print(f"\n3. 画质抽样统计（随机 {args.quality_n} 张）")
    random.seed(0)
    sample = random.sample(files, min(args.quality_n, len(files)))
    bright, contrast, sharp = quality_metrics(sample)
    describe("亮度均值", bright)
    describe("对比度std", contrast)
    describe("清晰度", sharp)
    print("   -> 亮度、对比度、清晰度分布都很宽，说明拍摄背景/光照/清晰度差异明显，"
          "这也是为什么需要归一化和数据增强。")

    print("\n4. 出图")
    fig_overview(files, out / "data_overview.png")
    print(f"   {out / 'data_overview.png'}")
    src = random.Random(1).choice(cats)
    fig_preprocess(src, out / "preprocess_preview.png")
    print(f"   {out / 'preprocess_preview.png'}  (样例 {src.name})")
    fig_augment(src, out / "augment_preview.png")
    print(f"   {out / 'augment_preview.png'}  (样例 {src.name})")


if __name__ == "__main__":
    main()
