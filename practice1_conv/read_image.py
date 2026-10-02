"""实践一 · 第 1 步：把一张图片拆成数字。

打印：
  1. Pillow 读到的原始信息（格式 / 颜色模式 / 尺寸）
  2. NumPy 数组（HWC）的 shape、dtype、最大值、最小值、一个像素的 RGB
  3. PyTorch Tensor（CHW）的同一组信息，并对照 transforms.ToTensor()

路径按脚本自身位置解析，所以在任何工作目录下运行都可以：
    python read_image.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR / "images" / "iam_nailong.png"


def main():
    # ---------- 1. Pillow 读图 ----------
    img_raw = Image.open(IMG_PATH)
    print("=== 1. Pillow 读到的原始信息 ===")
    print("文件格式 format:", img_raw.format)
    print("颜色模式 mode  :", img_raw.mode)
    print("尺寸 size(宽,高):", img_raw.size)

    img = img_raw.convert("RGB")          # 统一成 3 通道，丢掉 alpha
    print("convert('RGB') 后 mode:", img.mode)

    # ---------- 2. NumPy 数组：HWC ----------
    arr = np.array(img)
    print("\n=== 2. NumPy 数组（HWC）===")
    print("shape:", arr.shape, "  ->  (高 H, 宽 W, 通道 C)")
    print("dtype:", arr.dtype)
    print("min  :", arr.min(), "   max:", arr.max())
    print("第 1 个像素（左上角）RGB arr[0, 0]:", arr[0, 0])

    h, w, c = arr.shape
    print(f"像素总数 {h} x {w} = {h * w}，每个像素 {c} 个通道，"
          f"共 {h * w * c} 个 0~255 的整数")

    # ---------- 3. PyTorch Tensor：CHW ----------
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    print("\n=== 3. PyTorch Tensor（CHW，from_numpy + permute）===")
    print("shape:", tuple(tensor.shape), "  ->  (通道 C, 高 H, 宽 W)")
    print("dtype:", tensor.dtype)
    print("min  :", tensor.min().item(), "   max:", tensor.max().item())
    print("同一个像素 tensor[:, 0, 0]:", tensor[:, 0, 0].tolist())

    # ---------- 4. 对照：transforms.ToTensor() ----------
    t_float = transforms.ToTensor()(img)
    print("\n=== 4. 对照 transforms.ToTensor() ===")
    print("shape:", tuple(t_float.shape), "  dtype:", t_float.dtype)
    print("min  : %.4f   max: %.4f" % (t_float.min().item(), t_float.max().item()))
    print("同一个像素: [%.4f, %.4f, %.4f]" % tuple(t_float[:, 0, 0].tolist()))
    print("\n结论：两条路得到的 shape 相同(都是 CHW)，区别只在 dtype/取值范围——"
          "from_numpy 是纯搬运(uint8, 0~255)，ToTensor 额外做了 float32 和 /255。")


if __name__ == "__main__":
    main()
