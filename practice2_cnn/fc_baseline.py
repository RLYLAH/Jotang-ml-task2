"""趁热打铁第 1 题：用全连接层（MLP）实现同一任务，和 CNN 对比差距。

用法（工作目录 = practice2_cnn）：
    python fc_baseline.py --data_root data_split_sub2000 --epochs 15 --hidden 512 --seed 42

必须和 CNN 用**完全一样**的数据划分、batch、epoch、优化器和学习率，
只把网络结构从"卷积"换成"全连接"，差距才有意义。
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_loaders, INPUT_SIZE
from model import SimpleCNN
# 直接复用 CNN 训练脚本里的训练/评估函数，保证两边流程完全一致
from train import evaluate, train_one_epoch, set_seed, get_device


class MLP(nn.Module):
    """把 3x128x128 直接拉平成 49152 维再全连接。"""

    def __init__(self, hidden=512, num_classes=1, dropout=0.3):
        super().__init__()
        in_size = 3 * INPUT_SIZE * INPUT_SIZE
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_split_sub2000")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="fc_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device} | 全连接基线 hidden={args.hidden} | epochs={args.epochs}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=bool(args.augment),
    )

    model = MLP(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_cnn = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
    print(f"全连接基线参数量: {n_params:,}")
    print(f"CNN 参数量      : {n_cnn:,}")
    print(f"参数量倍数      : {n_params / n_cnn:.1f} 倍")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0
    epoch_times = []
    best_path = os.path.join(args.out_dir, f"best_{args.tag}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        epoch_times.append(dt)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"[{epoch:02d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    te_loss, te_acc, te_cm = evaluate(model, test_loader, criterion, device)
    print(f"\n全连接基线：best val acc={best_val_acc:.4f}（epoch {best_epoch}），"
          f"test acc={te_acc:.4f}")
    print(f"  平均每个 epoch {sum(epoch_times) / len(epoch_times):.1f}s")
    print(f"  test 混淆矩阵 [真实][预测] = {te_cm.tolist()}")

    # 曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("MLP Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, history["train_acc"], label="train")
    axes[1].plot(ep, history["val_acc"], label="val")
    axes[1].set_title("MLP Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"curves_{args.tag}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(args.out_dir, f"log_{args.tag}.txt"), "w", encoding="utf-8") as f:
        f.write(f"model: MLP(Flatten -> Linear({3*INPUT_SIZE*INPUT_SIZE},{args.hidden}) -> ReLU "
                f"-> Dropout -> Linear({args.hidden},1))\n")
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"params: {n_params}\n")
        f.write(f"cnn_params: {n_cnn}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_acc: {best_val_acc:.4f}\n")
        f.write(f"test_acc: {te_acc:.4f}\n")
        f.write(f"test_cm: {te_cm.tolist()}\n")
        f.write(f"mean_epoch_time_s: {sum(epoch_times) / len(epoch_times):.2f}\n")
        for k, v in history.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
