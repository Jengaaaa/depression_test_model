import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==============================================
# HRV 기반 스트레스 예측 모델 (메인 모델)
# ==============================================

DATA_PATH = Path(__file__).resolve().parents[1] / "document_model" / "data" / "hrv_dataset" / "data" / "processed_hrv.csv"


# --------------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------------
def load_hrv_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ HRV 데이터셋이 없습니다: {path}")

    df = pd.read_csv(path)
    print(f"✅ HRV 데이터 로드 완료: {df.shape[0]} rows, {df.shape[1]} columns")

    # Stress Label 생성 (3단계 분류)
    df["stress_label"] = pd.qcut(
        df["SDNN"],  # SDNN 낮을수록 스트레스 높음
        q=3,
        labels=["high", "medium", "low"]
    )

    # 입력 피처
    features = ["MeanNN", "SDNN", "RMSSD", "pNN50", "LF", "HF", "LF_HF"]
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(df[features].mean())
    y = df["stress_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"📌 사용 피처: {features}")
    print(f"📌 클래스 분포:\n{y.value_counts()}\n")

    return X_scaled, y


# --------------------------------------------------------
# 2. 모델 학습 & K-Fold 평가
# --------------------------------------------------------
def train_hrv_model(X, y):
    print("=== HRV Stress Prediction Training (K-Fold) ===\n")

    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
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

    print("===============================================")
    print(f"🎯 평균 Accuracy: {np.mean(accs):.4f}")
    print(f"🎯 평균 F1 Score: {np.mean(f1s):.4f}")


# --------------------------------------------------------
# 3. 실행부
# --------------------------------------------------------
def main():
    print("=== HRV Stress Model Start ===")
    print(f"DATA_PATH : {DATA_PATH}")

    X, y = load_hrv_dataset(DATA_PATH)
    train_hrv_model(X, y)


if __name__ == "__main__":
    main()
