"""训练脚本。

用法（工作目录 = practice2_cnn）：
    # 主实验（全量数据 + 默认增强）
    python train.py --data_root data_split --epochs 20 --tag full_aug --seed 42

    # 对照实验：关掉数据增强
    python train.py --data_root data_split --epochs 15 --tag subset_noaug --augment 0

    # 改进实验：强化增强 + 更长训练
    python train.py --data_root data_split --epochs 25 --tag full_aug_v2 --strong_aug 1

产物（outputs/）：
    best_{tag}.pt        验证集准确率最高的权重
    curves_{tag}.png     loss / accuracy 曲线
    log_{tag}.txt        超参、逐 epoch 指标、最终 test 混淆矩阵
"""

import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_loaders
from model import SimpleCNN


def set_seed(seed):
    """固定随机种子，让对照实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """返回 (平均 loss, 准确率, 混淆矩阵)。混淆矩阵下标为 [真实][预测]，0=cat, 1=dog。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    cm = np.zeros((2, 2), dtype=int)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        gt = labels.squeeze(1).long()

        correct += (preds == gt).sum().item()
        total += imgs.size(0)
        for g, p in zip(gt.cpu().tolist(), preds.cpu().tolist()):
            cm[g][p] += 1

    return total_loss / total, correct / total, cm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        correct += (preds == labels.squeeze(1).long()).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def plot_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_split")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", type=int, default=1, help="1=开增强, 0=关增强（对照实验）")
    parser.add_argument("--strong_aug", type=int, default=0, help="1=用强化增强（改进实验）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="cnn")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = get_device()
    print(f"使用设备: {device} | seed={args.seed} | augment={args.augment} "
          f"| strong_aug={args.strong_aug} | epochs={args.epochs}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=bool(args.augment),
        strong=bool(args.strong_aug),
    )

    model = SimpleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
    best_path = os.path.join(args.out_dir, f"best_{args.tag}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time() - t0:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  -> 保存新最优模型到 {best_path} (val_acc={best_val_acc:.4f})")

    print(f"\n训练完成，最优 val acc = {best_val_acc:.4f}（第 {best_epoch} 个 epoch）")

    # 只加载一次最优权重，在 val / test 上各评一次
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    va_loss, va_acc, va_cm = evaluate(model, val_loader, criterion, device)
    te_loss, te_acc, te_cm = evaluate(model, test_loader, criterion, device)
    print(f"最优模型 val : loss={va_loss:.4f}, acc={va_acc:.4f}, cm={va_cm.tolist()}")
    print(f"最优模型 test: loss={te_loss:.4f}, acc={te_acc:.4f}, cm={te_cm.tolist()}")
    print("  混淆矩阵下标 [真实][预测]，0=cat, 1=dog")
    print(f"  test 错分：cat->dog {te_cm[0][1]} 张，dog->cat {te_cm[1][0]} 张")

    plot_curves(history, os.path.join(args.out_dir, f"curves_{args.tag}.png"))

    gap = history["train_acc"][best_epoch - 1] - best_val_acc
    with open(os.path.join(args.out_dir, f"log_{args.tag}.txt"), "w", encoding="utf-8") as f:
        f.write(f"tag: {args.tag}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"augment: {args.augment}\n")
        f.write(f"strong_aug: {args.strong_aug}\n")
        f.write(f"params: {n_params}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_acc: {best_val_acc:.4f}\n")
        f.write(f"train_acc_at_best_epoch: {history['train_acc'][best_epoch - 1]:.4f}\n")
        f.write(f"train_val_gap_at_best_epoch: {gap:.4f}\n")
        f.write(f"val_acc_final: {va_acc:.4f}\n")
        f.write(f"test_loss: {te_loss:.4f}\n")
        f.write(f"test_acc: {te_acc:.4f}\n")
        f.write(f"test_cm: {te_cm.tolist()}\n")
        f.write(f"test_cat_to_dog: {int(te_cm[0][1])}\n")
        f.write(f"test_dog_to_cat: {int(te_cm[1][0])}\n")
        for k, v in history.items():
            f.write(f"{k}: {v}\n")

    print(f"曲线保存至 {args.out_dir}/curves_{args.tag}.png")


if __name__ == "__main__":
    main()
