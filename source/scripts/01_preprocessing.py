# -*- coding: utf-8 -*-
"""
01 数据预处理：特征缩放 + 类别不平衡处理（SMOTE 过采样）

输出：
    data/processed_train.csv   缩放 + SMOTE 后的训练集
    data/processed_test.csv    缩放后的测试集（保持原始分布，用于公平评估）
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "creditcard.csv"

FEATURE_COLS = None  # 动态确定：除 Time/Amount/Class 外均为 V1~V28


def load_data() -> pd.DataFrame:
    if not RAW.exists():
        raise FileNotFoundError(
            f"未找到数据集 {RAW}，请先运行 python scripts/download_data.py"
        )
    return pd.read_csv(RAW)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Amount 缩放至 [0,1]，Time 由秒转换为小时。"""
    df = df.copy()
    df["Time_hours"] = df["Time"] / 3600.0  # 秒 -> 小时
    df["Amount_scaled"] = MinMaxScaler().fit_transform(df[["Amount"]])
    df.drop(columns=["Time", "Amount"], inplace=True)
    return df


def split_and_smote(df: pd.DataFrame):
    """划分训练/测试集，并在训练集上应用 SMOTE。"""
    global FEATURE_COLS
    FEATURE_COLS = [c for c in df.columns if c != "Class"]

    X = df[FEATURE_COLS]
    y = df["Class"]

    # 分层划分，保证测试集保留真实的类别不平衡分布
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("SMOTE 前训练集分布:", dict(y_train.value_counts()))
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("SMOTE 后训练集分布:", dict(pd.Series(y_train_res).value_counts()))

    return X_train_res, y_train_res, X_test, y_test


def main() -> None:
    df = load_data()
    print(f"原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")
    print("欺诈占比: {:.4%}".format(df["Class"].mean()))

    df = preprocess(df)
    X_tr, y_tr, X_te, y_te = split_and_smote(df)

    train_df = pd.DataFrame(X_tr, columns=FEATURE_COLS)
    train_df["Class"] = y_tr.values
    test_df = pd.DataFrame(X_te, columns=FEATURE_COLS)
    test_df["Class"] = y_te.values

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "processed_train.csv", index=False)
    test_df.to_csv(DATA_DIR / "processed_test.csv", index=False)
    print("已输出 data/processed_train.csv 与 data/processed_test.csv")


if __name__ == "__main__":
    main()
