# 信用卡欺诈检测（Credit Card Fraud Detection）

数据挖掘实训项目：基于 SMOTE 过采样、随机森林特征工程与多模型对比的信用卡欺诈检测。

- **任务**：二分类（正常交易 = 0，欺诈交易 = 1），预测交易是否欺诈
- **数据集**：Kaggle 信用卡欺诈公开数据集（`creditcard.csv`），特征 V1~V28（PCA 处理）+ Time + Amount，标签 Class
- **评价指标**：F1 分数、AUC-ROC、AUC-PR、混淆矩阵

## 目录结构

```
credit-fraud-detection/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_data.py            # Kaggle 数据下载
│   ├── 01_preprocessing.py         # 特征缩放 + SMOTE 过采样
│   ├── 02_feature_engineering.py   # 随机森林特征重要性 + t-SNE 降维
│   ├── 03_model_training.py        # 逻辑回归 / 随机森林训练
│   └── 04_evaluation.py            # 混淆矩阵 + ROC/PR + 阈值调优
└── data/                           # 数据目录（不入库，见下方说明）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 下载数据（需要 Kaggle 账号与 API Key，或手动放置 creditcard.csv 到 data/）
python scripts/download_data.py

# 2. 依次执行流水线
python scripts/01_preprocessing.py
python scripts/02_feature_engineering.py
python scripts/03_model_training.py
python scripts/04_evaluation.py
```

## 数据处理说明

- **Amount**：特征缩放至 [0,1]
- **Time**：单位为秒，转换为小时
- **V1~V28**：已 PCA 处理，无需额外缩放
- **类别不平衡**：使用 SMOTE 合成少数类过采样技术，使正负样本接近后训练

## 数据上传 GitHub 说明

原始 `creditcard.csv` 约 150MB，超过 GitHub 单文件 100MB 限制，推荐二选一：

1. **Git LFS**（推荐）：仓库根目录已有 `.gitattributes`，执行 `git lfs install` 后
   `git add data/*.csv` 即可正常追踪。
2. **仅提交采样**：运行
   `python scripts/download_data.py --sample 20000` 生成约 20,000 行的采样数据
   （约 8MB）用于演示与轻量复现。

## 结果要点（来自实验报告）

- 阈值 0.1 → 0.9：召回率从 0.9836 持续下降至 0.9014
- 阈值 0.6：AUC-PR 达到峰值 0.97525，综合性能最优
- 所有阈值下 AUC-PR 均在 0.92 以上，模型识别能力优秀
- 欺诈交易金额呈现"小而散"特征；9:00–23:00 为消费高频时段
- V8、V13、V15、V20~V28 等变量无法区分类别，予以剔除
