import pandas as pd
from pathlib import Path

# ===============================
# HRV + 수면 데이터 통합 스크립트
# ===============================

BASE_DIR = Path(__file__).resolve().parents[1] / "document_model" / "data"
HRV_PATH = BASE_DIR / "hrv_dataset" / "data" / "processed_hrv.csv"
SLEEP_PATH = BASE_DIR / "sleep_dataset" / "processed_sleep.csv"
OUT_PATH = BASE_DIR / "merged_dataset.csv"

def main():
    print("=== Dataset Merge Start ===")
    print(f"HRV_PATH: {HRV_PATH}")
    print(f"SLEEP_PATH: {SLEEP_PATH}")
    print(f"OUT_PATH: {OUT_PATH}")

    # 1️⃣ 데이터 로드
    df_hrv = pd.read_csv(HRV_PATH)
    df_sleep = pd.read_csv(SLEEP_PATH)
    print(f"HRV shape: {df_hrv.shape}, Sleep shape: {df_sleep.shape}")

    # 2️⃣ HRV와 수면 데이터 ID/피험자 기준 정리
    df_hrv["id"] = df_hrv["id"].astype(str).str.lower()
    if "gender" in df_sleep.columns:
        df_sleep["gender"] = df_sleep["gender"].str.lower()

    # 3️⃣ 결합 키가 없으므로 단순 병합 (가장 일반적인 방식)
    # 수면 데이터의 평균값을 HRV에 병합
    numeric_cols = df_sleep.select_dtypes(include=["number"]).columns
    df_sleep_mean = df_sleep[numeric_cols].mean().to_frame().T
    df_merged = df_hrv.copy()
    for col in df_sleep_mean.columns:
        df_merged[col] = df_sleep_mean[col].values[0]

    # 4️⃣ 스트레스 라벨 생성 (조건 조정 가능)
    if "label" in df_hrv.columns:
        df_merged["stress_label"] = df_hrv["label"].apply(lambda x: 1 if "stress" in str(x).lower() else 0)
    else:
        df_merged["stress_label"] = 0  # fallback

    # 5️⃣ 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ Saved merged dataset → {OUT_PATH}")
    print(df_merged.head())

if __name__ == "__main__":
    main()
