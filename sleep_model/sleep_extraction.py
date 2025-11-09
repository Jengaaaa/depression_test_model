import pandas as pd
from pathlib import Path

# ===============================
# Sleep 데이터 전처리 스크립트
# ===============================

BASE_DIR = Path(__file__).resolve().parents[1] / "document_model" / "data" / "sleep_dataset"
RAW_PATH = BASE_DIR / "Sleep_health_and_lifestyle_dataset.csv"
OUT_PATH = BASE_DIR / "processed_sleep.csv"

def main():
    print("=== Sleep Feature Extraction Start ===")
    print(f"Resolved RAW_PATH: {RAW_PATH}")
    print(f"Resolved OUT_PATH: {OUT_PATH}")

    # 1️⃣ CSV 로드
    df = pd.read_csv(RAW_PATH)
    print(f"원본 shape: {df.shape}")
    print("컬럼 목록:", list(df.columns))

    # 2️⃣ 컬럼명 정리
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # 3️⃣ 필요없는 항목 제거 (이 부분은 햄이 원하면 조정 가능)
    drop_cols = ["person_id"] if "person_id" in df.columns else []
    df = df.drop(columns=drop_cols, errors="ignore")

    # 4️⃣ 결측치 처리
    df = df.dropna(subset=["sleep_duration", "quality_of_sleep"], how="any")

    # 5️⃣ 문자열 → 숫자 변환
    mapping = {
        "Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4,
        "Insufficient": 1, "Normal": 2, "Sufficient": 3
    }
    df.replace(mapping, inplace=True)

    # 6️⃣ 파생 지표 (수면 효율 등 계산)
    df["sleep_efficiency"] = df["quality_of_sleep"] / df["sleep_duration"]
    df["stress_risk"] = (df["physical_activity_level"] / (df["sleep_duration"] + 1)) * 10

    # 7️⃣ 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ Saved processed sleep dataset → {OUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()
