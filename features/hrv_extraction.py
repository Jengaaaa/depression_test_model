import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm

# ============================================
# HRV Feature Extraction (SWELL, elapsedtime 기반 + 다중 시트)
# ============================================

# document_model/data/hrv_dataset/data 기준 경로
BASE_DIR = Path(__file__).resolve().parents[1] / "document_model" / "data" / "hrv_dataset" / "data"
RRI_DIR = BASE_DIR / "raw" / "rri"
LABEL_FILE = BASE_DIR / "raw" / "labels" / "hrv stress labels.xlsx"
OUT_PATH = BASE_DIR / "processed_hrv.csv"


# --------------------------------------------------------
# 1) RRI 파일 로드
#    - 각 파일: timestamp / rri(ms) 형태 2컬럼
#    - RRI를 300~2000ms 사이로 필터링 + 보간
#    - 분석 편의를 위해 time_s = 0,1,2,... 로 재정의
# --------------------------------------------------------
def load_rri_files(rri_dir: Path):
    data_dict = {}

    if not rri_dir.exists():
        raise FileNotFoundError(f"❌ RRI_DIR not found: {rri_dir}")

    for fp in sorted(rri_dir.glob("*.txt")):
        pid = fp.stem.lower()  # p1, p2, ...

        try:
            arr = np.loadtxt(fp)
            # 형상이 (N,2)가 아니면 스킵
            if arr.ndim != 2 or arr.shape[1] != 2:
                print(f"⚠️ Skipping {pid}: invalid shape {arr.shape}")
                continue

            df = pd.DataFrame(arr, columns=["raw_time", "rri_ms"])

            # RRI 정상 범위만 사용 (artifact 제거)
            df = df[(df["rri_ms"] >= 300) & (df["rri_ms"] <= 2000)]

            # 결측이 있으면 보간
            df["rri_ms"] = df["rri_ms"].interpolate()

            # elapsedtime 과 맞추기 위해 0,1,2,... 로 재정의
            df["time_s"] = np.arange(len(df))

            data_dict[pid] = df

        except Exception as e:
            print(f"⚠️ Failed to load {pid}: {e}")
            continue

    print(f"✅ Loaded {len(data_dict)} participant RRI files.")
    return data_dict


# --------------------------------------------------------
# 2) 라벨 엑셀 모든 시트 로드
#    - PP1 ~ PP25 등 여러 시트를 전부 합침
#    - subject → id (pp1 → p1)
#    - elapsedtime → elapsed (초 단위 숫자)
# --------------------------------------------------------
def load_all_labels(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"❌ Label file not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    all_rows = []

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)

        # 컬럼명 정규화
        df.columns = [c.strip().lower() for c in df.columns]

        # 주요 컬럼 rename (엑셀 구조에 따라 다를 수 있음, 필요 시 조정)
        df = df.rename(
            columns={
                "subject": "id",
                "elapsedtime": "elapsed",
                "condition": "condition",
                "label": "label",
            }
        )

        # id를 p1, p2 형태로 맞추기 (PP1 → p1)
        if "id" in df.columns:
            df["id"] = (
                df["id"]
                .astype(str)
                .str.lower()
                .str.replace("pp", "p", regex=False)
                .str.strip()
            )

        # elapsedtime → 숫자형으로 변환
        if "elapsed" in df.columns:
            df["elapsed"] = (
                df["elapsed"]
                .astype(str)
                .str.strip()
                .str.replace(" ", "", regex=False)
            )
            df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")

        all_rows.append(df)

    labels = pd.concat(all_rows, ignore_index=True)

    # 유효한 id / elapsed / condition 만 필터링
    labels = labels[
        labels["id"].notna()
        & labels["elapsed"].notna()
        & labels["condition"].notna()
    ]

    print(f"📌 Loaded {len(labels)} total label rows from {len(xls.sheet_names)} sheets.")
    return labels


# --------------------------------------------------------
# 3) HRV 피처 계산 함수
#    - Time Domain: MeanNN, SDNN, RMSSD, pNN50
#    - Freq Domain: LF, HF, LF/HF (Welch PSD)
# --------------------------------------------------------
def compute_hrv_features(rr_intervals: np.ndarray) -> dict:
    rr = np.array(rr_intervals, dtype=float)
    diff = np.diff(rr)
    feats = {}

    # --- Time domain ---
    feats["MeanNN"] = float(np.mean(rr))
    feats["SDNN"] = float(np.std(rr))
    feats["RMSSD"] = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else np.nan
    feats["pNN50"] = float(np.sum(np.abs(diff) > 50) / len(diff) * 100) if len(diff) > 0 else np.nan

    # --- Frequency domain ---
    try:
        f, psd = signal.welch(rr - np.mean(rr), fs=4.0, nperseg=min(256, len(rr)))
        lf_power = np.trapezoid(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
        hf_power = np.trapezoid(psd[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)])
        feats["LF"] = float(lf_power)
        feats["HF"] = float(hf_power)
        feats["LF_HF"] = float(lf_power / (hf_power + 1e-6))
    except Exception:
        feats["LF"], feats["HF"], feats["LF_HF"] = np.nan, np.nan, np.nan

    return feats


# --------------------------------------------------------
# 4) elapsedtime 기반으로 HRV 세그먼트 추출
#    - 참가자(id)별로 label/condition 그룹핑
#    - 각 그룹의 elapsed min~max를 하나의 구간으로 보고
#      RRI의 time_s 와 매칭해서 HRV 피처 계산
# --------------------------------------------------------
def extract_hrv_with_labels(rri_dict: dict, labels: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    print("\n=== Label Sample ===")
    print(labels.head())

    for pid, df_rri in tqdm(rri_dict.items(), desc="Processing participants"):
        # 해당 참가자 라벨만
        sub = labels[labels["id"] == pid]
        if sub.empty:
            continue

        # label + condition 기준으로 구간 묶기
        # (예: rest-R, neutral-N, time pressure-T, interrupt-I 등)
        grouped = sub.groupby(["label", "condition"])

        for (lbl, cond), g in grouped:
            start_t = g["elapsed"].min()
            end_t = g["elapsed"].max()

            # elapsedtime(초) 기준으로 RRI time_s 매칭
            seg = df_rri[(df_rri["time_s"] >= start_t) & (df_rri["time_s"] <= end_t)]
            if len(seg) < 10:  # 너무 짧은 구간은 스킵
                continue

            feats = compute_hrv_features(seg["rri_ms"].values)
            feats["Participant"] = pid
            feats["Label"] = lbl
            feats["Condition"] = cond
            feats["Start_elapsed"] = start_t
            feats["End_elapsed"] = end_t

            all_rows.append(feats)

    df_out = pd.DataFrame(all_rows)
    print(f"\n✅ Total extracted segments: {len(df_out)}")
    return df_out


# --------------------------------------------------------
# 5) 메인 실행부
# --------------------------------------------------------
def main():
    print("=== HRV Feature Extraction Start ===")
    print(f"RRI_DIR     : {RRI_DIR}")
    print(f"LABEL_FILE  : {LABEL_FILE}")
    print(f"OUTPUT_PATH : {OUT_PATH}")

    # 입력 파일 체크
    if not RRI_DIR.exists() or not LABEL_FILE.exists():
        print("❌ Required input files not found.")
        print(f"  - RRI_DIR   exists? {RRI_DIR.exists()}")
        print(f"  - LABEL_FILE exists? {LABEL_FILE.exists()}")
        raise SystemExit(1)

    # 1) RRI & Label 로드
    rri_dict = load_rri_files(RRI_DIR)
    labels = load_all_labels(LABEL_FILE)

    # 2) HRV 세그먼트 추출
    df_features = extract_hrv_with_labels(rri_dict, labels)

    # 3) CSV 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ HRV feature file saved → {OUT_PATH}")
    print("=== Sample preview ===")
    print(df_features.head(10))


if __name__ == "__main__":
    main()
