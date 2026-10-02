"""手写二维卷积（只依赖 NumPy，不调用任何现成卷积实现）。

  禁用：cv2.filter2D、scipy.signal.convolve2d、torch.nn.Conv2d 等
  本文件只用 Python 循环 + NumPy 的切片/逐元素乘法求和完成核心卷积过程。

实现细节：

1) 这里做的是**互相关（cross-correlation）**，不是数学定义上的卷积。
   数学卷积要先把这个核上下、左右各翻转一次，而本实现（以及 PyTorch 的
   nn.Conv2d、绝大多数深度学习框架）直接用核原样滑窗。因为 CNN 里的核是
   学出来的，翻不翻转只等价于把权重换个写法，效果完全一样，所以框架统一
   采用互相关并仍叫它 "conv"。

2) padding="same" 用 pad = K // 2 实现，**只在奇数卷积核 + stride=1 时**
   才能真正保持尺寸不变：
       K=3, stride=1 -> 输出尺寸不变
       K=5, stride=1 -> 输出尺寸不变
       K=4          -> pad=2，输出反而变大 1（严格 same 需要左右不对称填充）
       stride=2     -> 输出约缩小一半（same 语义只保证"补够了边"，
                       并不保证尺寸不变）
"""

import numpy as np


def conv2d_numpy(image, kernel, padding=0, stride=1):
    """
    image: H x W（灰度）或 H x W x C（彩色）的 numpy 数组
    kernel: kH x kW 的二维卷积核，对每个输入通道使用同一个核
    padding: 0 / 整数 表示每边补 padding 圈 0；"same" 表示补 K // 2 圈
    stride: 步长

    返回：2 维输入 -> (H_out, W_out)；3 维输入 -> (H_out, W_out, C)
    输出尺寸公式：H_out = (H + 2P - K) // S + 1
    """
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    # 灰度图补一维成 (H, W, 1)，统一走后面的三分支逻辑
    if image.ndim == 2:
        image = image[:, :, None]
        squeeze = True
    else:
        squeeze = False

    H, W, C = image.shape
    kH, kW = kernel.shape

    if padding == "same":
        pad_h, pad_w = kH // 2, kW // 2
    else:
        pad_h = pad_w = int(padding)

    if pad_h > 0 or pad_w > 0:
        image_pad = np.pad(
            image,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        image_pad = image

    Hp, Wp, _ = image_pad.shape

    out_h = (Hp - kH) // stride + 1
    out_w = (Wp - kW) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"卷积核比图像还大：输入 {(H, W, C)} kernel {(kH, kW)} "
            f"padding {padding} stride {stride}"
        )

    out = np.zeros((out_h, out_w, C), dtype=np.float32)

    # 滑动窗口：逐位置取区域 -> 逐位相乘 -> 求和
    for y in range(out_h):
        for x in range(out_w):
            y0, x0 = y * stride, x * stride
            region = image_pad[y0:y0 + kH, x0:x0 + kW, :]
            # kernel[:, :, None] 把 (kH,kW) 广播成 (kH,kW,1)，
            # 让同一个核对 C 个通道同时生效
            out[y, x, :] = np.sum(region * kernel[:, :, None], axis=(0, 1))

    if squeeze:
        out = out[:, :, 0]

    return out


if __name__ == "__main__":
    # 自检：用一个 3x3 单位核和最简输入验证公式与实现一致
    img = np.array([[[10., 20., 30.],
                     [40., 50., 60.],
                     [70., 80., 90.]]], dtype=np.float32)  # 1x3x3 灰度
    img = img[0]
    k = np.array([[0., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 0.]], dtype=np.float32)
    print("valid 输出:", conv2d_numpy(img, k, padding=0, stride=1))
    print("same  输出:", conv2d_numpy(img, k, padding="same", stride=1))
    print("灰度输入 (3,3) 的 same 输出形状:", conv2d_numpy(img, k, "same", 1).shape)
