"""独立推理程序：输入一张本地图片 -> 输出 猫/狗、预测概率、预处理后的图片。

用法（工作目录 = practice2_cnn）：
    python predict.py --ckpt outputs/best_full_aug.pt --img dc/test/1000.jpg

关键点：这里的预处理必须和训练时**完全一致**（dataset.eval_transform：
Resize(144) -> CenterCrop(128) -> ToTensor -> Normalize），否则同一张图在
训练和推理时会得到不同的输入分布，预测会失准。

产物：
    outputs/predict.png            左：原图，右：真正送进网络的 128x128 图
    outputs/predict_processed.png  单独保存预处理后的图片
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from PIL import Image

from dataset import eval_transform, MEAN, STD, INPUT_SIZE
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


def denormalize(t):
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(STD) + np.array(MEAN)
    return np.clip(x, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/predict.png")
    parser.add_argument("--proc_out", type=str, default="outputs/predict_processed.png")
    args = parser.parse_args()
    setup_cjk_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    img = Image.open(args.img).convert("RGB")
    print(f"输入图片 : {args.img}  原始尺寸(宽,高)={img.size} mode={img.mode}")
    print(f"预处理   : Resize(144)（短边缩放到 144，保持长宽比）-> "
          f"CenterCrop({INPUT_SIZE}) -> ToTensor -> Normalize")

    x = eval_transform(img).unsqueeze(0).to(device)
    print(f"送进网络的张量: shape={tuple(x.shape)} dtype={x.dtype} "
          f"值域[{x.min():.2f}, {x.max():.2f}]（已按 ImageNet 均值方差归一化）")

    with torch.no_grad():
        logit = model(x)
        prob_dog = torch.sigmoid(logit).item()

    is_dog = prob_dog > 0.5
    label_cn = "狗" if is_dog else "猫"
    label_en = "dog" if is_dog else "cat"
    prob_final = prob_dog if is_dog else 1 - prob_dog

    print("-" * 52)
    print(f"预测结果 : {label_cn}（{label_en}）")
    print(f"预测概率 : {prob_final:.4f}   [p(dog)={prob_dog:.4f}, p(cat)={1 - prob_dog:.4f}]")
    print("-" * 52)

    # 单独保存"真正送进网络的那张图"
    proc_np = denormalize(x[0])
    Path(args.proc_out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((proc_np * 255).astype(np.uint8)).save(args.proc_out)
    print(f"预处理后的图片保存到 {args.proc_out}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(img)
    axes[0].set_title(f"Input {img.size[0]}x{img.size[1]}")
    axes[0].axis("off")
    axes[1].imshow(proc_np)
    axes[1].set_title(f"Preprocessed {INPUT_SIZE}x{INPUT_SIZE}\n"
                      f"Pred: {label_cn}/{label_en}  {prob_final:.2%}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"可视化保存到 {args.out}")


if __name__ == "__main__":
    main()
