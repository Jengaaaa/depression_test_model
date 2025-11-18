import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

# =============================================
# SWELL HRV Sliding Window Extraction
# =============================================

BASE_DIR = Path(__file__).resolve().parents[1] / "document_model" / "data" / "hrv_dataset" / "data"
RRI_DIR = BASE_DIR / "raw" / "rri"
LABEL_PATH = BASE_DIR / "raw" / "labels" / "hrv stress labels.xlsx"
OUT_PATH = BASE_DIR / "processed_hrv_windowed.csv"

WINDOW_SIZE = 60     # 60초 창
WINDOW_STEP = 30     # 30초 이동


# --------------------------------------------------------
# 1) 모든 시트 라벨 로드
# --------------------------------------------------------
def load_all_labels(path):
    print("📌 Loading ALL label sheets...")
    xls = pd.ExcelFile(path)

    df_all = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df.columns = [c.strip().lower() for c in df.columns]

        # column 표준화
        df.rename(columns={
            "subject": "id",
            "elapsedtime": "elapsed",
            "label": "label",
            "condition": "condition",
        }, inplace=True)

        df["id"] = df["id"].astype(str).str.lower()
        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)
    print(f"📌 Loaded {len(df_all)} total label rows from {len(xls.sheet_names)} sheets.")

    return df_all


# --------------------------------------------------------
# 2) RRI 파일 로드
# --------------------------------------------------------
def load_rri_files(rri_dir):
    data = {}

    for fp in sorted(rri_dir.glob("*.txt")):
        pid = fp.stem.lower()
        arr = np.loadtxt(fp)

        if arr.ndim == 1 or arr.shape[1] != 2:
            print(f"⚠️ Skipping {pid}: Invalid shape {arr.shape}")
            continue

        df = pd.DataFrame(arr, columns=["time_s", "rri_ms"])

        # 이상치 제거 + 보간
        df = df[(df["rri_ms"] >= 300) & (df["rri_ms"] <= 2000)]
        df["rri_ms"] = df["rri_ms"].interpolate()

        # 상대 시간으로 재설정
        df["time_s"] = np.arange(len(df))

        data[pid] = df

    print(f"✅ Loaded {len(data)} participants RRI")
    return data


# --------------------------------------------------------
# 3) HRV feature 계산
# --------------------------------------------------------
def compute_hrv(rr):
    rr = np.array(rr)
    diff = np.diff(rr)

    features = {
        "MeanNN": rr.mean(),
        "SDNN": rr.std(),
        "RMSSD": np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else 0,
        "pNN50": np.mean(np.abs(diff) > 50) * 100,
    }

    # Welch PSD
    f, psd = welch(rr - rr.mean(), fs=4.0, nperseg=min(256, len(rr)))

    lf = np.trapezoid(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapezoid(psd[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)])

    features["LF"] = lf
    features["HF"] = hf
    features["LF_HF"] = lf / (hf + 1e-6)

    return features


# --------------------------------------------------------
# 4) 슬라이딩 윈도우 + 라벨 매칭
# --------------------------------------------------------
def extract_window_hrv(rri_dict, df_label):
    rows = []

    for pid, df_rri in tqdm(rri_dict.items(), desc="participants"):
        sub_labels = df_label[df_label["id"] == pid]

        if sub_labels.empty:
            print(f"⚠️ No labels for participant {pid}")
            continue

        max_time = df_rri["time_s"].max()

        for start in range(0, max_time - WINDOW_SIZE, WINDOW_STEP):
            end = start + WINDOW_SIZE

            seg = df_rri[(df_rri["time_s"] >= start) & (df_rri["time_s"] < end)]
            if len(seg) < 30:
                continue

            # 윈도우 중앙에 맞는 라벨 가져오기
            center_t = (start + end) / 2
            label_rows = sub_labels.iloc[(sub_labels["elapsed"] - center_t).abs().argsort()[:1]]

            label = label_rows["label"].iloc[0]
            cond = label_rows["condition"].iloc[0]

            feats = compute_hrv(seg["rri_ms"].values)
            feats.update({
                "Participant": pid,
                "Label": label,
                "Condition": cond,
                "Start": start,
                "End": end,
            })
            rows.append(feats)

    df_out = pd.DataFrame(rows)
    print(f"\n🎯 Total extracted rows: {len(df_out)}")
    return df_out


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("=== HRV Sliding Window Extraction Start ===")
    print(f"RRI_DIR : {RRI_DIR}")
    print(f"LABEL_PATH : {LABEL_PATH}")

    # Load All
    df_label = load_all_labels(LABEL_PATH)
    rri_dict = load_rri_files(RRI_DIR)

    df_out = extract_window_hrv(rri_dict, df_label)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n🎉 Saved → {OUT_PATH}")
    print(df_out.head())


if __name__ == "__main__":
    main()
