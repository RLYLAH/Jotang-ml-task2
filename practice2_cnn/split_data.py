"""
将 dc/train 下的猫狗图片按 70/15/15 分层抽样，整理成 ImageFolder 结构：
data_split/
├── train/{cat,dog}/
├── val/{cat,dog}/
└── test/{cat,dog}/
"""

import os
import argparse
import random
import shutil
from pathlib import Path

# 默认路径按本脚本位置解析，换机器/换目录都不用改代码
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = BASE_DIR / "dc" / "train"
DEFAULT_DST = BASE_DIR / "data_split"


def link_or_copy(src, dst, method="hardlink"):
    """优先硬链接，失败则复制。硬链接速度快且不占额外空间。"""
    if dst.exists():
        return
    if method == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default=str(DEFAULT_SRC),
        help="原始有标签 train 文件夹（默认 practice2_cnn/dc/train）"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(DEFAULT_DST),
        help="输出文件夹（默认 practice2_cnn/data_split）"
    )
    parser.add_argument(
        "--subset", type=int, default=0,
        help="每类使用多少张；0 表示全部使用。先跑 2000 快速验证。"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method", choices=["copy", "hardlink"], default="hardlink"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)

    assert src.exists(), f"源目录不存在: {src}"

    # 收集文件
    cats = sorted([p for p in src.iterdir() if p.is_file() and p.name.startswith("cat.")])
    dogs = sorted([p for p in src.iterdir() if p.is_file() and p.name.startswith("dog.")])
    print(f"找到 cat: {len(cats)} 张, dog: {len(dogs)} 张")

    # 是否用小子集
    if args.subset > 0:
        n = args.subset
        cats = random.sample(cats, min(n, len(cats)))
        dogs = random.sample(dogs, min(n, len(dogs)))
        print(f"[子集模式] 每类取 {n} 张 -> cat {len(cats)}, dog {len(dogs)}")

    # 分层抽样
    def split_list(lst):
        random.shuffle(lst)
        n = len(lst)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    cat_tr, cat_va, cat_te = split_list(cats)
    dog_tr, dog_va, dog_te = split_list(dogs)

    splits = {
        "train": {"cat": cat_tr, "dog": dog_tr},
        "val":   {"cat": cat_va, "dog": dog_va},
        "test":  {"cat": cat_te, "dog": dog_te},
    }

    # 创建目录
    for split_name in splits:
        for cls in ["cat", "dog"]:
            (dst / split_name / cls).mkdir(parents=True, exist_ok=True)

    # 拷贝/硬链接
    for split_name, cls_dict in splits.items():
        for cls, files in cls_dict.items():
            out_dir = dst / split_name / cls
            for i, f in enumerate(files):
                link_or_copy(f, out_dir / f.name, method=args.method)
            print(f"{split_name}/{cls}: {len(files)} 张")

    print("\n完成！目录结构：")
    print(dst)
    for split_name in splits:
        total = sum(len(v) for v in splits[split_name].values())
        print(f"  {split_name}/ 共 {total} 张")


if __name__ == "__main__":
    main()