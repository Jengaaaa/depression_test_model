import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


# === 절대경로 자동 설정 ===
BASE_DIR = Path(__file__).resolve().parents[1] / "document_model" / "data" / "sleep_dataset"
RAW_PATH = BASE_DIR / "Sleep_health_and_lifestyle_dataset.csv"
OUT_PATH = BASE_DIR / "processed_sleep.csv"


# --------------------------------------------------------
# 1️⃣ 데이터 로드
# --------------------------------------------------------
def load_sleep_data(csv_path: Path) -> pd.DataFrame:
    """Sleep Health & Lifestyle 데이터 로드"""
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ 데이터 로드 완료: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------------
# 2️⃣ 데이터 클리닝
# --------------------------------------------------------
def clean_sleep_data(df: pd.DataFrame) -> pd.DataFrame:
    """결측치, 이상치 처리 및 컬럼 정리"""
    df = df.copy()

    # 컬럼명 표준화
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # 결측값 제거
    df = df.dropna(subset=["sleep_duration", "quality_of_sleep", "stress_level", "physical_activity_level", "heart_rate"])

    # 이상치 제거
    df = df[(df["sleep_duration"] > 0) & (df["sleep_duration"] <= 12)]
    df = df[(df["quality_of_sleep"] >= 1) & (df["quality_of_sleep"] <= 10)]
    df = df[(df["stress_level"] >= 1) & (df["stress_level"] <= 10)]
    df = df[(df["physical_activity_level"] >= 0) & (df["physical_activity_level"] <= 100)]
    df = df[(df["heart_rate"] >= 40) & (df["heart_rate"] <= 130)]

    print(f"✅ 클리닝 완료: {df.shape[0]} rows 남음")
    return df


# --------------------------------------------------------
# 3️⃣ 피처 엔지니어링 (Stress 관련 파생 피처 생성)
# --------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """수면·활동·심박 기반 스트레스 민감도 피처 및 수면 효율 지표 생성"""

    df = df.copy()

    # === 정규화 (0~1 범위) ===
    df["sleep_norm"] = df["sleep_duration"] / df["sleep_duration"].max()
    df["quality_norm"] = df["quality_of_sleep"] / 10
    df["activity_norm"] = df["physical_activity_level"] / 100
    df["hr_norm"] = (df["heart_rate"] - df["heart_rate"].min()) / (df["heart_rate"].max() - df["heart_rate"].min())

    # === 스트레스 방향 반전 (값이 낮을수록 스트레스↑) ===
    df["sleep_stress"] = 1 - df["sleep_norm"]
    df["quality_stress"] = 1 - df["quality_norm"]
    df["activity_stress"] = 1 - df["activity_norm"]
    df["hr_stress"] = df["hr_norm"]

    # === 스트레스 지수 계산 (가중 평균) ===
    df["stress_index"] = (
        df["sleep_stress"] * 0.3 +
        df["quality_stress"] * 0.25 +
        df["activity_stress"] * 0.2 +
        df["hr_stress"] * 0.25
    ) * 100

    # === 수면 효율 지표 (Sleep Efficiency) ===
    df["sleep_efficiency"] = (df["sleep_duration"] / 8) * df["quality_of_sleep"]

    # === 스트레스 밸런스 지표 ===
    df["stress_balance"] = df["physical_activity_level"] / (df["stress_level"] + 1e-6)

    print("✅ 피처 엔지니어링 완료")
    return df


# --------------------------------------------------------
# 4️⃣ 메인 파이프라인
# --------------------------------------------------------
def main():
    print("=== Sleep Feature Extraction Start ===")
    print(f"RAW_PATH     : {RAW_PATH}")
    print(f"OUTPUT_PATH  : {OUT_PATH}")

    # 로드 & 전처리
    df_raw = load_sleep_data(RAW_PATH)
    df_clean = clean_sleep_data(df_raw)
    df_feat = feature_engineering(df_clean)

    # 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 전처리 완료 → {OUT_PATH}")
    print("=== Sample Preview ===")
    print(df_feat.head(10))


if __name__ == "__main__":
    main()
