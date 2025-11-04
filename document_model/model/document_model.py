# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# ✅ 상대경로 수정 (model 폴더에서 한 단계 위로 이동)
path = r"../data/archive (1)/Deepression.csv"

print("Path to dataset file:", os.path.abspath(path))

# 상위 디렉토리 파일 확인
print("\n📁 Files inside dataset directory:")
dir_path = os.path.dirname(path)
if os.path.exists(dir_path):
    for file in os.listdir(dir_path):
        print("-", file)
else:
    print("⚠️ 디렉토리를 찾을 수 없습니다:", dir_path)

# CSV 파일 로드
if os.path.exists(path):
    df = pd.read_csv(path, encoding="utf-8")
    print("\n✅ CSV 파일 로드 성공!")
    print(df.head())
else:
    print("\n❌ CSV 파일을 찾을 수 없습니다. 실제 파일 경로를 확인해주세요.")

# 데이터프레임 정보 출력
if 'df' in locals():
    print("\n📊 데이터프레임 정보:")
    print(df.info())
    print("\n데이터프레임 요약 통계:")
    print(df.describe())
    print("\n데이터프레임 컬럼명:")
    print(df.columns)   
    print("\n데이터프레임 크기:")
    print(df.shape) 
    print("\n데이터프레임 결측치 확인:")
    print(df.isnull().sum())
    print("\n데이터프레임 중복 행 확인:")
    print(df.duplicated().sum())
    print("\n데이터프레임의 처음 5개 행:")
    print(df.head())



# 컬럼명 정리 및 결측치 제거
df.columns = df.columns.str.strip()
df = df.dropna()

# X, y 분리
X = df.drop(columns=["Number", "Depression State"])
y = df["Depression State"]

print("✅ 결측치 제거 후:", df.shape)

# 🎯 라벨 문자열 정제
df["Depression State"] = (
    df["Depression State"]
    .astype(str)
    .str.strip()
    .str.replace(r"^\d+\s*", "", regex=True)  # 숫자 + 공백 제거
)

# 🎯 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Depression State"] = le.fit_transform(df["Depression State"])

print("\n🎯 정제 후 인코딩 매핑 결과:")
for i, label in enumerate(le.classes_):
    print(f"  {i}: {label}")

# X: 입력 피처(증상 데이터)
# y: 타깃(우울 상태)
X = df.drop(columns=["Depression State"])
y = df["Depression State"]

# train/test 80:20 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("학습 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb_model = XGBClassifier(random_state=42, eval_metric="mlogloss")


# ✅ 하이퍼파라미터 그리드 정의
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# ✅ GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,                # 5-Fold 교차검증
    n_jobs=-1,           # 모든 CPU 코어 사용
    verbose=2
)

print("\n🚀 Grid Search 시작 중...")
grid_search.fit(X_train, y_train)

# ✅ 결과 출력
print("\n✅ Grid Search 완료!")
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# ✅ 최적 모델로 테스트 데이터 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📈 Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
