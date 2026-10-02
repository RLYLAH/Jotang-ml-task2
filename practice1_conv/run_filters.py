"""实践一 · 第 2~4 步：用四种卷积核处理同一张图片，并比较 padding / stride 的影响
运行（任意工作目录）：
    python run_filters.py

输出（全部落在 practice1_conv/outputs/）：
    original.jpg            原图（保持原始尺寸，不做缩放）
    mean_blur.jpg           3x3 均值模糊
    gauss_blur.jpg          5x5 高斯模糊
    sharpen.jpg             锐化
    sobel_x.jpg              Sobel-X（竖直边缘）
    sobel_y.jpg              Sobel-Y（水平边缘）
    sobel_edge.jpg          灰度下 gx/gy 合成的边缘强度（黑底白线）
    sobel_edge_rgb.jpg      逐通道合成后全局归一化的版本（会出现彩色边缘）
    gray_mean_blur.jpg      灰度图分支演示
    comparison.png          2x4 对比图


"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # 只存图，不弹窗，方便脚本化运行
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from conv2d import conv2d_numpy

BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR / "images" / "iam_nailong.png"
OUT_DIR = BASE_DIR / "outputs"


# ---------------------------------------------------------------- 工具函数
def gaussian_kernel(size=5, sigma=1.0):
    """生成 size x size 的二维高斯核，总和归一化为 1。"""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def to_uint8(x):
    """像素语义的结果：裁剪到 0~255 再转 uint8。"""
    return np.clip(x, 0, 255).astype(np.uint8)


def normalize_to_uint8(x):
    """响应值语义的结果（边缘强度、特征图）：线性拉伸到 0~255 再转 uint8。"""
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo > 1e-8:
        x = (x - lo) / (hi - lo) * 255.0
    else:
        x = np.zeros_like(x)
    return x.astype(np.uint8)


def out_size(h, w, k, padding, stride):
    """输出尺寸公式：(H + 2P - K) // S + 1"""
    p = k // 2 if padding == "same" else int(padding)
    return (h + 2 * p - k) // stride + 1, (w + 2 * p - k) // stride + 1


# ---------------------------------------------------------------- 主流程
def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 1. 读图（保持原始尺寸）
    img = Image.open(IMG_PATH).convert("RGB")
    img_np = np.array(img).astype(np.float32)
    H, W, C = img_np.shape
    print("=" * 68)
    print("1. 输入")
    print(f"   原图 mode={img.mode} size(宽,高)={img.size} -> 数组 shape={img_np.shape} "
          f"(H={H}, W={W}, C={C})")
    print(f"   注意：本脚本不做 resize，下面的尺寸都和 read_image.py 对上。")

    # 2. 卷积核
    mean_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    gauss_kernel = gaussian_kernel(size=5, sigma=1.0)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    print("\n2. 使用的卷积核")
    for name, k in [("均值 3x3", mean_kernel), ("高斯 5x5", gauss_kernel),
                    ("锐化 3x3", sharpen_kernel), ("Sobel-X", sobel_x),
                    ("Sobel-Y", sobel_y)]:
        print(f"   {name}（核内和 = {k.sum():.3f}）")
        print("      " + np.array2string(k, precision=4, suppress_small=False).replace("\n", "\n      "))

    # 3. 灰度图分支：验证手写卷积函数对 (H, W) 二维输入同样可用
    gray = np.array(img.convert("L")).astype(np.float32)
    gray_same = conv2d_numpy(gray, mean_kernel, padding="same", stride=1)
    gray_valid = conv2d_numpy(gray, mean_kernel, padding=0, stride=1)
    print("\n3. 灰度图分支（Image.convert('L')）")
    print(f"   灰度输入 shape={gray.shape}（只有 H、W，没有通道维）")
    print(f"   conv2d_numpy 输出：valid={gray_valid.shape}  same={gray_same.shape}")
    print("   -> 二维进二维出，说明灰度图是 1 个通道；彩色图是 3 个通道，输出多一维 C。")

    # 4. 彩色图卷积（same padding，保持尺寸）
    t0 = time.perf_counter()
    mean_img = conv2d_numpy(img_np, mean_kernel, padding="same")
    gauss_img = conv2d_numpy(img_np, gauss_kernel, padding="same")
    sharp_img = conv2d_numpy(img_np, sharpen_kernel, padding="same")
    gx = conv2d_numpy(img_np, sobel_x, padding="same")
    gy = conv2d_numpy(img_np, sobel_y, padding="same")
    t_conv = time.perf_counter() - t0
    print(f"\n4. 彩色图 5 次 same 卷积耗时 {t_conv:.2f}s（每次约 {t_conv / 5:.2f}s）")
    print(f"   输出 shape：均值{mean_img.shape} 高斯{gauss_img.shape} "
          f"锐化{sharp_img.shape} gx{gx.shape} gy{gy.shape}")

    # 5. 合成边缘：两种做法
    edge_rgb = np.sqrt(gx ** 2 + gy ** 2)                      # 逐通道算模长（结果仍是 3 通道）
    g_gray = np.array(img.convert("L")).astype(np.float32)
    gx_g = conv2d_numpy(g_gray, sobel_x, padding="same")
    gy_g = conv2d_numpy(g_gray, sobel_y, padding="same")
    edge_gray = np.sqrt(gx_g ** 2 + gy_g ** 2)                 # 灰度下算模长（单通道）
    print("\n5. 边缘合成")
    print(f"   逐通道版 edge_rgb shape={edge_rgb.shape}，数值范围 "
          f"[{edge_rgb.min():.1f}, {edge_rgb.max():.1f}]")
    print(f"   灰度版   edge_gray shape={edge_gray.shape}，数值范围 "
          f"[{edge_gray.min():.1f}, {edge_gray.max():.1f}]")
    print("   -> 逐通道版三个通道的缩放不同，全局归一化后会显示成彩色边缘；")
    print("      想得到课本上的黑白边缘图，应先转灰度再合成。")

    # 6. 保存图片
    Image.fromarray(img_np.astype(np.uint8)).save(OUT_DIR / "original.jpg")
    Image.fromarray(to_uint8(mean_img)).save(OUT_DIR / "mean_blur.jpg")
    Image.fromarray(to_uint8(gauss_img)).save(OUT_DIR / "gauss_blur.jpg")
    Image.fromarray(to_uint8(sharp_img)).save(OUT_DIR / "sharpen.jpg")
    Image.fromarray(normalize_to_uint8(gx)).save(OUT_DIR / "sobel_x.jpg")
    Image.fromarray(normalize_to_uint8(gy)).save(OUT_DIR / "sobel_y.jpg")
    Image.fromarray(normalize_to_uint8(edge_gray)).save(OUT_DIR / "sobel_edge.jpg")
    Image.fromarray(normalize_to_uint8(edge_rgb)).save(OUT_DIR / "sobel_edge_rgb.jpg")
    Image.fromarray(to_uint8(gray_same)).save(OUT_DIR / "gray_mean_blur.jpg")
    print(f"\n6. 图片已保存到 {OUT_DIR}")

    # 7. 对比图（2 行 4 列）
    panels = [
        ("Original", img_np.astype(np.uint8), None),
        ("Mean Blur 3x3", to_uint8(mean_img), None),
        ("Gaussian Blur 5x5", to_uint8(gauss_img), None),
        ("Sharpen 3x3", to_uint8(sharp_img), None),
        ("Sobel X (vertical edges)", normalize_to_uint8(gx), None),
        ("Sobel Y (horizontal edges)", normalize_to_uint8(gy), None),
        ("Sobel Edge = sqrt(gx^2+gy^2), gray", normalize_to_uint8(edge_gray), "gray"),
        ("Grayscale branch: mean blur", to_uint8(gray_same), "gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison.png", dpi=120)
    plt.close(fig)
    print(f"   对比图：{OUT_DIR / 'comparison.png'}")

    # 8. valid vs same：尺寸 + 边缘像素的定量对比
    print("\n" + "=" * 68)
    print("7. valid（不填充）vs same（补 1 圈 0），3x3 均值核，stride=1")
    valid = conv2d_numpy(img_np, mean_kernel, padding=0, stride=1)
    same = conv2d_numpy(img_np, mean_kernel, padding="same", stride=1)
    print(f"   输入        ：{img_np.shape}")
    print(f"   valid 输出  ：{valid.shape}   （高 -{H - valid.shape[0]}，宽 -{W - valid.shape[1]}）")
    print(f"   same  输出  ：{same.shape}   （尺寸不变）")
    print("   边缘像素变化（各行/列的通道均值）：")
    print(f"     原图 顶行均值            {img_np[0].mean():8.2f}")
    print(f"     valid 顶行均值           {valid[0].mean():8.2f}   （由原图第 1~3 行算出，没有被 0 污染）")
    print(f"     same  顶行均值           {same[0].mean():8.2f}   （由原图第 1 行 + 一行 0 算出，被压暗）")
    print(f"     原图 底行均值            {img_np[-1].mean():8.2f}")
    print(f"     valid 底行均值           {valid[-1].mean():8.2f}")
    print(f"     same  底行均值           {same[-1].mean():8.2f}")
    print(f"     valid 内部均值           {valid[1:-1, 1:-1].mean():8.2f}")
    print(f"     same  内部均值           {same[1:-1, 1:-1].mean():8.2f}   （内部两者几乎一致）")
    print("   -> same 靠补 0 把边缘像素也纳入了窗口，代价是最外一圈输出被 0 拉低（变暗）；")
    print("      valid 不补 0、边缘更真实，但每卷一次就缩小 K-1 个像素。")

    # 9. 卷积核大小 / padding / stride 与输出尺寸的关系
    print("\n" + "=" * 68)
    print("8. 输出尺寸公式 H_out = (H + 2P - K) // S + 1 的实测表")
    print(f"   输入 (H,W) = ({H},{W})")
    print(f"   {'核K':>4} {'padding':>8} {'stride':>7} {'公式输出':>12} {'实际卷积输出':>16}")
    combos = [(3, 0, 1), (3, "same", 1), (3, 0, 2), (3, "same", 2),
              (5, 0, 1), (5, "same", 1), (7, "same", 1)]
    for k, p, s in combos:
        fh, fw = out_size(H, W, k, p, s)
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)
        real = conv2d_numpy(img_np, kernel, padding=p, stride=s)
        flag = "OK" if (real.shape[0], real.shape[1]) == (fh, fw) else "不一致!"
        print(f"   {k:>4} {str(p):>8} {s:>7} {str((fh, fw)):>12} {str(real.shape[:2]):>16}  {flag}")
    print("   结论：")
    print("     - 补 0 变多(P↑) 或 核变小(K↓) -> 输出变大；步长变大(S↑) -> 输出变小。")
    print("     - padding='same'(P=K//2) 只在奇数核 + stride=1 时严格保持尺寸。")
    print("     - stride=2 时 same 也只保证“边补够了”，输出仍是约 1/2（256->128 若不 crop）。")

    # 10. 手写卷积的耗时随图大小增长
    print("\n" + "=" * 68)
    print("9. 手写卷积耗时（纯 Python 双重循环，每次都是 3x3 均值核 same）")
    for side in (H, 256, 512):
        test = np.array(img.resize((side, side))).astype(np.float32)
        n = test.shape[0] * test.shape[1]
        t0 = time.perf_counter()
        conv2d_numpy(test, mean_kernel, padding="same")
        dt = time.perf_counter() - t0
        print(f"   {test.shape[0]}x{test.shape[1]}（{n:>7} 个输出位置）: {dt:5.2f}s")
    print("   -> 位置数翻 4 倍，耗时也约翻 4 倍；换成大图必须向量化"
          "（如 np.lib.stride_tricks.sliding_window_view）才实用。")


if __name__ == "__main__":
    main()
