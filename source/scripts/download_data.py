# -*- coding: utf-8 -*-
"""
下载 Kaggle 信用卡欺诈数据集（creditcard.csv）。

两种方式：
1. 自动下载：需要 Kaggle 账号，首次运行提示输入 API Key（Kaggle -> Settings -> API -> Create New Token）
2. 手动放置：将 creditcard.csv 放到 data/ 目录下即可

可选参数：
    --sample N   仅抽取 N 行采样数据（默认全量下载）
"""
import argparse
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = DATA_DIR / "creditcard.csv"

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def download_full() -> None:
    """使用 kagglehub 下载数据集到 data/ 目录。"""
    try:
        import kagglehub
    except ImportError:
        print("缺少 kagglehub，请先执行: pip install kagglehub")
        sys.exit(1)

    print("正在从 Kaggle 下载数据集（约 150MB，需要网络与 Kaggle API Key）...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    src = Path(path) / "creditcard.csv"
    if not src.exists():
        # 部分版本压缩包内文件名可能不同，逐个查找
        cands = list(Path(path).rglob("*.csv"))
        if not cands:
            print("下载完成但未找到 CSV 文件，请检查下载目录:", path)
            sys.exit(1)
        src = cands[0]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(src), str(TARGET))
    print(f"数据已保存至 {TARGET}")


def download_sample(n: int) -> None:
    """下载后仅保留 n 行采样数据，用于轻量演示与 GitHub 提交。"""
    download_full()
    import pandas as pd
    df = pd.read_csv(TARGET)
    sampled = df.sample(n=n, random_state=42)
    out = DATA_DIR / "creditcard_sample.csv"
    sampled.to_csv(out, index=False)
    # 删除全量文件，避免误提交
    TARGET.unlink(missing_ok=True)
    print(f"采样数据已保存至 {out}（{len(sampled)} 行）")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载信用卡欺诈数据集")
    parser.add_argument("--sample", type=int, default=0,
                        help="仅生成 N 行采样数据（用于演示 / GitHub 提交）")
    args = parser.parse_args()

    if TARGET.exists():
        print(f"检测到本地已有数据: {TARGET}，跳过下载")
        if args.sample:
            import pandas as pd
            df = pd.read_csv(TARGET)
            out = DATA_DIR / "creditcard_sample.csv"
            df.sample(n=args.sample, random_state=42).to_csv(out, index=False)
            print(f"采样数据已保存至 {out}")
        return

    if args.sample:
        download_sample(args.sample)
    else:
        download_full()


if __name__ == "__main__":
    main()
