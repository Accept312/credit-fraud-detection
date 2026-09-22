# -*- coding: utf-8 -*-
"""
02 特征工程：随机森林特征重要性（Gini + Permutation）+ t-SNE 降维

输出：
    output/feature_importance.png    特征重要性排序图
    output/tsne_visualization.png    t-SNE 降维可视化
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# 报告中识别出无法区分类别的变量（可视化为依据，可结合重要性分数复核）
WEAK_FEATURES = [
    "V8", "V13", "V15", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28",
]


def load_processed():
    train = pd.read_csv(DATA_DIR / "processed_train.csv")
    test = pd.read_csv(DATA_DIR / "processed_test.csv")
    return train, test


def gini_importance(X, y) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1,
        random_state=42, class_weight="balanced_subsample",
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def permutation_importance(X, y):
    from sklearn.inspection import permutation_importance
    rf = RandomForestClassifier(
        n_estimators=60, max_depth=10, n_jobs=-1,
        random_state=42, class_weight="balanced_subsample",
    )
    rf.fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def tsne_plot(X_sample, y_sample) -> None:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
    X_2d = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    ax.scatter(X_2d[y_sample == 0, 0], X_2d[y_sample == 0, 1],
               s=4, alpha=0.5, label="正常交易", color="#0F766E")
    ax.scatter(X_2d[y_sample == 1, 0], X_2d[y_sample == 1, 1],
               s=6, alpha=0.9, label="欺诈交易", color="#C2410C")
    ax.set_title("t-SNE 降维可视化（信用卡交易）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tsne_visualization.png")
    plt.close(fig)
    print("已输出 output/tsne_visualization.png")


def main() -> None:
    train, test = load_processed()
    X = train.drop(columns=["Class"])
    y = train["Class"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gini = gini_importance(X, y)
    print("Gini 重要性 Top10:\n", gini.head(10).round(4).to_string())

    # 复核弱特征：确认它们的重要性排序靠后
    weak_rank = {f: int(np.where(gini.index == f)[0][0]) for f in WEAK_FEATURES if f in gini.index}
    print("弱特征排名（值越大越靠后）:", weak_rank)

    perm = permutation_importance(X, y)
    print("Permutation 重要性 Top10:\n", perm.head(10).round(4).to_string())

    # 特征重要性排序图（Gini）
    fig, ax = plt.subplots(figsize=(9, 10), dpi=130)
    top = gini.head(20)
    colors = ["#C2410C" if f in WEAK_FEATURES else "#0F766E" for f in top.index]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_title("随机森林特征重要性（Gini）")
    ax.set_xlabel("重要性")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png")
    plt.close(fig)
    print("已输出 output/feature_importance.png")

    # t-SNE：由于全量数据量大，抽 20,000 行做降维可视化
    sample = train.sample(n=min(20000, len(train)), random_state=42)
    tsne_plot(sample.drop(columns=["Class"]).values, sample["Class"].values)


if __name__ == "__main__":
    main()
