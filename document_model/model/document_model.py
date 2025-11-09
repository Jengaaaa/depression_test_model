# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

# ============================================================
# 1️⃣ 데이터 다운로드 및 로드
# ============================================================
path_kaggle = kagglehub.dataset_download("hamjashaikh/mental-health-detection-dataset")
print("✅ Kaggle 데이터 다운로드 경로:", path_kaggle)
print("📁 폴더 내 파일 목록:", os.listdir(path_kaggle))

# CSV 자동 탐색
csv_files = [f for f in os.listdir(path_kaggle) if f.endswith(".csv")]
print("✅ CSV 파일 목록:", csv_files)

csv_path = os.path.join(path_kaggle, csv_files[0])
df1 = pd.read_csv(csv_path)
df2 = df1.copy()

print("✅ 데이터 로드 완료")
print("파일명:", csv_files[0])
print("데이터 크기:", df1.shape)

# ============================================================
# 2️⃣ 컬럼 정제 및 병합
# ============================================================
common_cols = list(set(df1.columns) & set(df2.columns))
print("\n📊 공통 컬럼:", common_cols)

df = pd.concat([df1[common_cols], df2[common_cols]], axis=0, ignore_index=True)
df = df.dropna().drop_duplicates()
print(f"✅ 병합 및 정제 완료, shape: {df.shape}")

# ============================================================
# 3️⃣ 라벨 정제 및 인코딩
# ============================================================
label_col = 'Depression State'
df[label_col] = df[label_col].astype(str).str.strip().str.replace(r"[\t\n\r]", "", regex=True)
df[label_col] = df[label_col].str.replace(r"^[0-9]+", "", regex=True).str.strip()
df[label_col] = df[label_col].str.lower().replace({
    "no depression": "no_depression",
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe"
})

print("\n🎯 정제된 클래스 목록:", df[label_col].unique())

le = LabelEncoder()
df[label_col] = le.fit_transform(df[label_col])
print("✅ 인코딩 클래스:", list(le.classes_))

X = df.drop(columns=[label_col, 'Number '], errors='ignore')
y = df[label_col]

# ============================================================
# 4️⃣ 데이터 불균형 해결 (SMOTE)
# ============================================================
print("\n⚖️ SMOTE 오버샘플링 적용 중...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("✅ SMOTE 완료:", X_resampled.shape)

# ============================================================
# 5️⃣ 스케일링
# ============================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

# ============================================================
# 6️⃣ Train/Test 분리
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"\n📊 학습 데이터 크기: {X_train.shape}")
print(f"📊 테스트 데이터 크기: {X_test.shape}")

# ============================================================
# 7️⃣ 데이터 증강 (약간의 노이즈 추가)
# ============================================================
X_aug = X_train + np.random.normal(0, 0.03, X_train.shape)
y_aug = y_train.copy()
X_train_final = np.vstack([X_train, X_aug])
y_train_final = np.hstack([y_train, y_aug])
print(f"✅ 데이터 증강 완료: {X_train_final.shape}")

# ============================================================
# 8️⃣ XGBoost 하이퍼파라미터 튜닝
# ============================================================
print("\n🔍 GridSearchCV로 최적 파라미터 탐색 중...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
grid = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    param_grid, cv=3, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train_final, y_train_final)

print("🏆 Best Params:", grid.best_params_)
print("🔥 Best CV Accuracy:", grid.best_score_)

# ============================================================
# 9️⃣ 최적 XGBoost + 앙상블 학습
# ============================================================
best_xgb = XGBClassifier(**grid.best_params_, random_state=42, eval_metric='mlogloss')
rf = RandomForestClassifier(n_estimators=200, random_state=42)
lgb = LGBMClassifier(random_state=42)

voting = VotingClassifier(
    estimators=[('xgb', best_xgb), ('rf', rf), ('lgb', lgb)],
    voting='soft'
)

print("\n🚀 앙상블 모델 학습 중...")
voting.fit(X_train_final, y_train_final)

# ============================================================
# 🔟 평가
# ============================================================
y_pred = voting.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Test Accuracy: {acc:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

# ============================================================
# 🔁 교차검증
# ============================================================
scores = cross_val_score(voting, X_scaled, y_resampled, cv=5, scoring='accuracy')
print(f"\n🔁 5-Fold 교차검증 평균 정확도: {scores.mean():.4f}")

# ============================================================
# 🔍 Feature Importance 시각화
# ============================================================
best_xgb.fit(X_train_final, y_train_final)
plt.figure(figsize=(8,6))
plt.title("Feature Importance (XGBoost)")
plt.barh(X.columns, best_xgb.feature_importances_)
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
