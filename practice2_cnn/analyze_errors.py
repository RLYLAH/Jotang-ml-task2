"""错分样本分析：全量扫描 -> 混淆矩阵 -> 按类别均衡抽样查看。

用法（工作目录 = practice2_cnn）：
    python analyze_errors.py --ckpt outputs/best_full_aug.pt --split test --n 16

为什么不能用"从头扫到第 n 张错分就停"：
    ImageFolder 是按类名排序的（cat.* 全部排在 dog.* 前面），从头扫描时先遇到的
    必定是 cat 的错分样本，画出来会清一色是 "GT:cat Pred:dog"，而 dog->cat 的错误
    一张都看不到，很容易得出错误结论。所以这里先扫完整个 split，统计混淆矩阵，
    再从两类错误里各取一半。

产物：
    outputs/confusion_matrix.png   混淆矩阵（计入准确率、每类召回率）
    outputs/errors.png             GT/Pred/概率 标注的高置信错分样本
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from dataset import eval_transform, eval_transform_stretch, MEAN, STD
from model import SimpleCNN


def denormalize(t):
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(STD) + np.array(MEAN)
    return np.clip(x, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data_split")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--transform", type=str, default="crop",
                        choices=["crop", "stretch"],
                        help="crop=Resize(144)+CenterCrop（不拉伸）；"
                             "stretch=Resize((144,144))（历史版本，会拉伸）")
    parser.add_argument("--out", type=str, default="outputs/errors.png")
    parser.add_argument("--cm_out", type=str, default="outputs/confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    tf = eval_transform if args.transform == "crop" else eval_transform_stretch
    ds = datasets.ImageFolder(f"{args.data_root}/{args.split}", transform=tf)
    idx2cls = {v: k for k, v in ds.class_to_idx.items()}
    print(f"ckpt: {args.ckpt}")
    print(f"预处理: {args.transform}")
    print(f"{args.split} 集共 {len(ds)} 张，类别映射 {ds.class_to_idx}")

    # ---------- 1. 全量扫描 ----------
    probs = np.zeros(len(ds), dtype=np.float32)
    gts = np.zeros(len(ds), dtype=np.int64)
    with torch.no_grad():
        for i in range(len(ds)):
            img_t, label = ds[i]
            logit = model(img_t.unsqueeze(0).to(device))
            probs[i] = torch.sigmoid(logit).item()
            gts[i] = label

    preds = (probs > 0.5).astype(np.int64)
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gts, preds):
        cm[g][p] += 1

    acc = (cm[0][0] + cm[1][1]) / cm.sum()
    cat_recall = cm[0][0] / max(1, cm[0].sum())
    dog_recall = cm[1][1] / max(1, cm[1].sum())
    cat_prec = cm[0][0] / max(1, cm[:, 0].sum())
    dog_prec = cm[1][1] / max(1, cm[:, 1].sum())

    print("\n=== 混淆矩阵 [真实][预测]（0=cat, 1=dog）===")
    print(f"            预测cat  预测dog")
    print(f"  真实cat   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  真实dog   {cm[1][0]:6d}  {cm[1][1]:6d}")
    print(f"\n  总体准确率 : {acc:.4f}  ({cm[0][0] + cm[1][1]}/{cm.sum()})")
    print(f"  cat 召回率 : {cat_recall:.4f}    cat 精确率: {cat_prec:.4f}")
    print(f"  dog 召回率 : {dog_recall:.4f}    dog 精确率: {dog_prec:.4f}")
    print(f"  错分总数   : {cm[0][1] + cm[1][0]}")
    print(f"    cat -> dog: {cm[0][1]} 张")
    print(f"    dog -> cat: {cm[1][0]} 张")

    # ---------- 2. 阈值敏感性（看是否只是 0.5 这个阈值不合适） ----------
    print("\n=== 判决阈值扫描 ===")
    print(f"  {'阈值':>6} {'acc':>8} {'cat->dog':>9} {'dog->cat':>9}")
    for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        p = (probs > th).astype(np.int64)
        c = np.zeros((2, 2), dtype=int)
        for g, pp in zip(gts, p):
            c[g][pp] += 1
        a = (c[0][0] + c[1][1]) / c.sum()
        print(f"  {th:>6.2f} {a:>8.4f} {c[0][1]:>9d} {c[1][0]:>9d}")
    print("  概率分布: 均值 %.3f，中位数 %.3f" % (probs.mean(), np.median(probs)))
    print("  错分样本的 |p-0.5| 中位数: %.3f（越小说明模型越'不确定'）"
          % np.median(np.abs(probs[gts != preds] - 0.5)))

    # ---------- 3. 混淆矩阵图 ----------
    Path(args.cm_out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i][j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred cat", "pred dog"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true cat", "true dog"])
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title(f"Confusion matrix ({args.split})\nacc={acc:.4f}, n={cm.sum()}")
    plt.tight_layout()
    plt.savefig(args.cm_out, dpi=150)
    plt.close(fig)
    print(f"\n混淆矩阵图保存到 {args.cm_out}")

    # ---------- 4. 按类别均衡抽样展示高置信错分 ----------
    wrong = np.where(preds != gts)[0]
    if len(wrong) == 0:
        print("没有错分样本。")
        return

    half = max(1, args.n // 2)
    cat_wrong = wrong[gts[wrong] == 0]
    dog_wrong = wrong[gts[wrong] == 1]
    # "越自信越错"：cat->dog 按 p 从大到小，dog->cat 按 p 从小到大
    sel_cat = cat_wrong[np.argsort(-probs[cat_wrong])][:half]
    sel_dog = dog_wrong[np.argsort(probs[dog_wrong])][:half]
    sel = list(sel_cat) + list(sel_dog)

    print(f"\n从 {len(cat_wrong)} 张 cat->dog 与 {len(dog_wrong)} 张 dog->cat 中各取 "
          f"{len(sel_cat)}/{len(sel_dog)} 张（按模型自信程度排序）")

    cols = 4
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.1 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, sel):
        img_t, label = ds[i]
        ax.imshow(denormalize(img_t))
        pred = int(preds[i])
        p_correct = probs[i] if pred == 1 else 1 - probs[i]
        ax.set_title(f"GT:{idx2cls[label]} Pred:{idx2cls[pred]}\n"
                     f"p(dog)={probs[i]:.2f} conf={p_correct:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(sel):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"错分样本图保存到 {args.out}")

    print("\n最自信的 8 个错分样本（文件, 真实, 预测, p(dog)）：")
    order = wrong[np.argsort(-np.abs(probs[wrong] - 0.5))]
    for i in order[:8]:
        print(f"  {Path(ds.samples[i][0]).name:>16}  gt={idx2cls[gts[i]]:<3} "
              f"pred={idx2cls[preds[i]]:<3} p_dog={probs[i]:.3f}")


if __name__ == "__main__":
    main()
