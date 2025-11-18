import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ============================================
# HRV Stress Model (Sliding Window 387 samples)
# ============================================

DATA_PATH = Path(__file__).resolve().parents[1] / "document_model" / "data" / "hrv_dataset" / "data" / "processed_hrv_windowed.csv"


# --------------------------------------------------------
# 1) 데이터 로드
# --------------------------------------------------------
def load_hrv_window_dataset(path):
    df = pd.read_csv(path)
    print(f"✅ 데이터 로드 완료: {df.shape[0]} rows, {df.shape[1]} columns")

    # 피처 선택
    feature_cols = ["MeanNN", "SDNN", "RMSSD", "pNN50", "LF", "HF", "LF_HF"]

    X = df[feature_cols]
    y = df["Condition"]   # N(휴식), T(스트레스1), I(스트레스2) 등

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"📌 사용 피처: {feature_cols}")
    print(f"📌 클래스 분포:\n{y.value_counts()}")
    print()

    return X_scaled, y


# --------------------------------------------------------
# 2) 모델 학습 & 평가
# --------------------------------------------------------
def train_model(X, y):
    print("=== HRV Stress Prediction (Sliding Window) ===\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        accs.append(acc)
        f1s.append(f1)

        print(f"=== Fold {fold} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

    print("=======================================")
    print(f"🎯 평균 Accuracy: {np.mean(accs):.4f}")
    print(f"🎯 평균 F1 Score: {np.mean(f1s):.4f}")


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("=== HRV Stress Window Model Start ===")
    print(f"DATA_PATH: {DATA_PATH}")

    X, y = load_hrv_window_dataset(DATA_PATH)
    train_model(X, y)


if __name__ == "__main__":
    main()
