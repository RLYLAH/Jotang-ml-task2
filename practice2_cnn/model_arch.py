"""绘制模型结构图，并打印逐层形状/参数量表。

用法（工作目录 = practice2_cnn）：
    python model_arch.py

产物：outputs/model_arch.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from model import SimpleCNN


def setup_cjk_font():
    """matplotlib 默认字体没有中文，这里挑一个系统里可用的中文字体。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun",
                 "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def layer_table(model, x):
    """逐层前向，记录输出形状与参数量。"""
    rows = [("input", "输入图片", tuple(x.shape), 0)]
    h = x
    for i, block in enumerate(model.features):
        for j, layer in enumerate(block):
            h = layer(h)
            rows.append((f"features[{i}][{j}]", layer.__class__.__name__,
                         tuple(h.shape), sum(p.numel() for p in layer.parameters())))
    h = model.gap(h)
    rows.append(("gap", "AdaptiveAvgPool2d", tuple(h.shape),
                 sum(p.numel() for p in model.gap.parameters())))
    h = model.classifier(h)
    rows.append(("classifier", "Dropout+Linear", tuple(h.shape),
                 sum(p.numel() for p in model.classifier.parameters())))
    return rows


def draw(rows, out_path):
    # 把连续的同形状卷积层合并成一行展示，图才不会太长
    boxes = [
        ("Input\n3 x 128 x 128", "#e8f4ff"),
        ("Block 1\n[Conv 3x3, 3->32 + BN + ReLU] x2\nMaxPool 2x2\n-> 32 x 64 x 64", "#dff0d8"),
        ("Block 2\n[Conv 3x3, 32->64 + BN + ReLU] x2\nMaxPool 2x2\n-> 64 x 32 x 32", "#dff0d8"),
        ("Block 3\n[Conv 3x3, 64->128 + BN + ReLU] x2\nMaxPool 2x2\n-> 128 x 16 x 16", "#dff0d8"),
        ("Block 4\n[Conv 3x3, 128->256 + BN + ReLU] x2\nMaxPool 2x2\n-> 256 x 8 x 8", "#dff0d8"),
        ("AdaptiveAvgPool2d(1)\n-> 256 x 1 x 1", "#fcf3cf"),
        ("Flatten -> Dropout(0.3) -> Linear(256 -> 1)\n-> 1 个 logit（猫=0，狗=1）", "#fadbd8"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(boxes)
    top, bottom = 0.965, 0.035
    slot = (top - bottom) / n
    h_box = slot * 0.70

    for i, (text, color) in enumerate(boxes):
        y = top - slot * (i + 1) + slot * 0.15
        ax.add_patch(FancyBboxPatch((0.10, y), 0.80, h_box,
                                    boxstyle="round,pad=0.012",
                                    linewidth=1.4, edgecolor="#555555", facecolor=color))
        ax.text(0.50, y + h_box / 2, text, ha="center", va="center", fontsize=10.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.50, y), (0.50, y - slot * 0.30),
                                         arrowstyle="-|>", mutation_scale=16,
                                         linewidth=1.6, color="#555555"))

    ax.text(0.5, 0.995, "SimpleCNN（可训练参数 1,173,473）",
            ha="center", va="top", fontsize=12.5, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_cjk_font()
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    model = SimpleCNN()
    x = torch.randn(1, 3, 128, 128)

    rows = layer_table(model, x)
    print(f"{'位置':<18}{'层':<22}{'输出形状':<22}{'参数':>10}")
    print("-" * 74)
    for loc, name, shape, n_param in rows:
        print(f"{loc:<18}{name:<22}{str(shape):<22}{n_param:>10,}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 74)
    print(f"{'合计可训练参数':<62}{total:>10,}")

    out = out_dir / "model_arch.png"
    draw(rows, out)
    print(f"\n结构图保存到 {out}")


if __name__ == "__main__":
    main()
