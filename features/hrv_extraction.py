import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm

# ===============================
# HRV 데이터 전처리 스크립트 (최종 수정판)
# ===============================

# 현재 파일 기준으로 절대경로 자동 설정
BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "document_model" / "data" / "hrv_dataset" / "data"
RAW_DIR = BASE_DATA_DIR / "raw" / "rri"
LABEL_PATH = BASE_DATA_DIR / "raw" / "labels" / "hrv stress labels.xlsx"
OUT_PATH = BASE_DATA_DIR / "processed_hrv.csv"

# -------------------------------
# 1. RRI 텍스트 파일 로드
# -------------------------------
def load_rri_files(raw_dir):
    """RRI 텍스트 파일 로드 (각 피험자별로 길이 다름)
    - 파일 구조: 각 줄에 1개 이상의 공백 구분 숫자
    """
    raw_dir = Path(raw_dir)
    data = {}

    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {raw_dir}")

    for fp in raw_dir.iterdir():
        if fp.suffix.lower() == ".txt":
            pid = fp.stem
            text = fp.read_text(encoding="utf-8", errors="ignore")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue

            rr_values = []
            for ln in lines:
                # 한 줄에 여러 숫자가 있는 경우 split으로 나누기
                for token in ln.replace(",", ".").split():
                    try:
                        rr_values.append(float(token))
                    except ValueError:
                        continue

            if len(rr_values) == 0:
                print(f"⚠️ Skipping {pid}: no valid numeric RR values found.")
                continue

            rr_intervals = np.array(rr_values, dtype=float)
            data[pid] = rr_intervals

    if len(data) == 0:
        raise RuntimeError(f"No valid RRI data found under {raw_dir}")
    return data

# -------------------------------
# 2. HRV 지표 계산 함수
# -------------------------------
def compute_hrv_features(rr_intervals):
    """RRI 배열로부터 HRV 주요 지표 계산"""
    diff = np.diff(rr_intervals)
    features = {
        "MeanNN": np.mean(rr_intervals),
        "SDNN": np.std(rr_intervals),
        "RMSSD": np.sqrt(np.mean(diff**2)),
        "pNN50": np.sum(np.abs(diff) > 50) / len(diff) * 100 if len(diff) > 0 else np.nan,
    }
    # 주파수 기반 HRV (Welch)
    try:
        f, psd = signal.welch(rr_intervals, fs=4.0)
        lf_band = np.trapz(psd[(f >= 0.04) & (f < 0.15)])
        hf_band = np.trapz(psd[(f >= 0.15) & (f < 0.4)])
        features["LF_HF"] = lf_band / (hf_band + 1e-6)
    except Exception:
        features["LF_HF"] = np.nan
    return features

# -------------------------------
# 3. 라벨 병합
# -------------------------------
def merge_with_labels(hrv_dict, label_path):
    """Excel 라벨과 HRV 특징 병합"""
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # subject 컬럼을 ID로 통일
    df_labels = pd.read_excel(label_path)
    df_labels.rename(columns={"subject": "id"}, inplace=True)
    df_labels["id"] = df_labels["id"].astype(str)

    rows = []
    for pid, rri in tqdm(hrv_dict.items(), desc="Processing subjects"):
        feats = compute_hrv_features(rri)
        feats["id"] = pid
        rows.append(feats)

    df_hrv = pd.DataFrame(rows)
    df_merged = pd.merge(df_hrv, df_labels, on="id", how="inner")
    return df_merged

# -------------------------------
# 4. 메인 실행부
# -------------------------------
def main():
    print("=== HRV Feature Extraction Start ===")
    print(f"Resolved RAW_DIR: {RAW_DIR}")
    print(f"Resolved LABEL_PATH: {LABEL_PATH}")
    print(f"Resolved OUT_PATH: {OUT_PATH}")

    # 경로 확인
    if not RAW_DIR.exists():
        print(f"❌ RAW_DIR does not exist: {RAW_DIR}")
        raise SystemExit(1)
    if not LABEL_PATH.exists():
        print(f"❌ LABEL_PATH does not exist: {LABEL_PATH}")
        raise SystemExit(1)

    # 처리 시작
    hrv_dict = load_rri_files(RAW_DIR)
    df_final = merge_with_labels(hrv_dict, LABEL_PATH)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved processed HRV dataset → {OUT_PATH}")
    print("=== Sample preview ===")
    print(df_final.head())

# -------------------------------
# 5. 실행
# -------------------------------
if __name__ == "__main__":
    main()
