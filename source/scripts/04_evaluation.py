# -*- coding: utf-8 -*-
"""
04 模型评估：混淆矩阵 + ROC/PR 曲线 + 阈值调优

输出：
    output/confusion_matrix.png    混淆矩阵
    output/pr_curve.png            Precision-Recall 曲线（多阈值）
    output/threshold_tuning.csv    不同阈值下的召回率 / 精确率 / AUC-PR
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_processed():
    train = pd.read_csv(DATA_DIR / "processed_train.csv")
    test = pd.read_csv(DATA_DIR / "processed_test.csv")
    return train, test


def main() -> None:
    train, test = load_processed()
    X_train, y_train = train.drop(columns=["Class"]), train["Class"]
    X_test, y_test = test.drop(columns=["Class"]), test["Class"]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1,
        random_state=42, class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 混淆矩阵（默认阈值 0.5）
    cm = confusion_matrix(y_test, (y_score >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_xticks([0, 1], ["正常", "欺诈"])
    ax.set_yticks([0, 1], ["正常", "欺诈"])
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13)
    ax.set_title("混淆矩阵（默认阈值 0.5）")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close(fig)
    print("已输出 output/confusion_matrix.png")

    # 2) ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    ax.plot(fpr, tpr, color="#0F766E", lw=2,
            label=f"ROC (AUC = {roc_auc_score(y_test, y_score):.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#B45309", lw=1)
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC 曲线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curve.png")
    plt.close(fig)
    print("已输出 output/roc_curve.png")

    # 3) PR 曲线（多阈值标注）
    prec, rec, thr = precision_recall_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.plot(rec, prec, color="#C2410C", lw=2,
            label=f"PR (AUC = {auc(rec, prec):.4f})")
    for t in [0.1, 0.6, 0.9]:
        idx = np.argmin(np.abs(thr - t))
        ax.scatter(rec[idx], prec[idx], color="#0F766E", zorder=3)
        ax.annotate(f"阈值 {t}", (rec[idx], prec[idx]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("Precision-Recall 曲线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curve.png")
    plt.close(fig)
    print("已输出 output/pr_curve.png")

    # 4) 阈值调优表：不同阈值下的召回率 / 精确率 / AUC-PR
    rows = []
    pr_auc_all = auc(rec, prec)  # 整体 AUC-PR 基线
    for t in THRESHOLDS:
        y_pred = (y_score >= t).astype(int)
        rows.append({
            "threshold": t,
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
        })
    tuning = pd.DataFrame(rows)
    tuning.to_csv(OUT_DIR / "threshold_tuning.csv", index=False)
    print("已输出 output/threshold_tuning.csv")
    print(f"\n整体 AUC-PR（基线参考）: {pr_auc_all:.4f}")
    print("\n阈值调优结果:")
    print(tuning.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
