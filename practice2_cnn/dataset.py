"""DataLoader 与预处理流水线。

目录结构（由 split_data.py 生成）：
    data_split/{train,val,test}/{cat,dog}/*.jpg

三条流水线的区别：
    train_transform         训练集用，带随机增强
    strong_train_transform  改进实验用的强化版（翻转 + 旋转 + 更激进裁剪 + 随机擦除）
    eval_transform          验证/测试/推理用，**不做任何随机增强**

为什么验证集/测试集不能做随机增强：增强会改变样本的分布，同一张图每次评估
得到不同输入，指标就不可复现；而且裁剪/旋转可能切掉关键部位、改变类别线索，
测出来的分数就不再代表真实泛化能力。训练集用增强是为了**制造更多样的训练样本、
抑制过拟合**，这两件事的目的不同。
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet 统计量：本任务数据量不大，用这套通用的 mean/std 足够
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 网络输入边长（模型里按 128x128 搭的）
INPUT_SIZE = 128


def build_train_transform(strong=False):
    """训练集增强。strong=True 为改进实验用的强化版本。"""
    if strong:
        ops = [
            transforms.Resize((176, 176)),
            # scale 下探到 0.5：允许更大范围的裁剪，逼模型别只盯住主体中心
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),          # 小幅旋转，模拟拍摄角度差异
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ]
        post = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            # 随机擦除：遮掉一小块，缓解模型依赖个别显著部位（如鼻子、耳朵）
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value="random"),
        ]
    else:
        ops = [
            transforms.Resize((160, 160)),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        post = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    return transforms.Compose(ops + post)


# 训练集默认流水线
train_transform = build_train_transform(strong=False)
# 改进实验用的强化增强
strong_train_transform = build_train_transform(strong=True)

# 验证 / 测试 / 推理：只做确定性的缩放 + 中心裁剪 + 归一化
#
# 注意这里是 Resize(144)（只写一个数字）而不是 Resize((144,144))：
#   外层卷积要求输入尺寸固定，但直接把任意长宽比的图拉成正方形会**拉伸**图像
#   （本数据集宽高比实测 0.3 ~ 5.9，500x374 的图会被横向压扁 25%）。
#   Resize(144) 表示"把短边缩放到 144、长边按比例缩放"，再 CenterCrop(128)
#   取中心，物体形状不会被改变。这与 torchvision 官方 ImageNet 流程
#   （Resize(256) -> CenterCrop(224)）一致。
eval_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 历史版本：直接拉成 144x144 的正方形，会产生拉伸。
# 单独保留一份，用于对比"拉伸带来的精度损失"（analyze_errors.py --transform stretch）。
eval_transform_stretch = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_loaders(data_root, batch_size=32, num_workers=4, augment=True, strong=False):
    """
    返回 train_loader, val_loader, test_loader。

    augment=False  训练集也不做增强（走 eval_transform），用于对照实验。
    strong=True    训练集用强化版增强（改进实验用）。

    batch 是怎么组成的：DataLoader 每次取 batch_size 张已预处理好的图，
    堆叠成 (N, C, H, W) = (32, 3, 128, 128) 的 float32 张量，标签堆成 (32,)。
    训练集 shuffle=True 打乱顺序，避免同一类的样本总排在一起影响梯度；
    drop_last=True 丢掉最后一个不满 32 的 batch，让 BatchNorm 每批统计量更稳。
    """
    if strong and augment:
        train_tf = strong_train_transform
    elif augment:
        train_tf = train_transform
    else:
        # 无增强对照：直接复用 eval 流水线，保证"唯一变量是增强"
        train_tf = eval_transform

    train_set = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_set = datasets.ImageFolder(f"{data_root}/val", transform=eval_transform)
    test_set = datasets.ImageFolder(f"{data_root}/test", transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"类别映射: {train_set.class_to_idx}    # cat=0, dog=1")
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    print(f"一个 batch 的形状: ({batch_size}, 3, {INPUT_SIZE}, {INPUT_SIZE})")
    return train_loader, val_loader, test_loader
