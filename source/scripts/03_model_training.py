# -*- coding: utf-8 -*-
"""
03 模型构建：逻辑回归 / 随机森林多模型对比训练

输出：
    output/models_compare.csv    各模型在测试集（原始不平衡分布）上的评估指标
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_processed():
    train = pd.read_csv(DATA_DIR / "processed_train.csv")
    test = pd.read_csv(DATA_DIR / "processed_test.csv")
    return train, test


def evaluate(y_true, y_score) -> dict:
    # 以 0.5 为默认阈值得出混淆矩阵相关指标
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def main() -> None:
    train, test = load_processed()
    X_train, y_train = train.drop(columns=["Class"]), train["Class"]
    X_test, y_test = test.drop(columns=["Class"]), test["Class"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=-1,
            random_state=42, class_weight="balanced_subsample",
        ),
    }

    rows = []
    for name, model in models.items():
        print(f"训练 {name} ...")
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_score)
        metrics["model"] = name
        rows.append(metrics)
        print(f"  {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "model"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(OUT_DIR / "models_compare.csv")
    print("已输出 output/models_compare.csv")


if __name__ == "__main__":
    main()
