import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 스트레스 예측 모델 학습 (최종 수정판)
# ===============================

DATA_PATH = Path(__file__).resolve().parents[1] / "document_model" / "data" / "merged_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "stress_model.pt"

# -------------------------------
# 모델 정의
# -------------------------------
class StressModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # ✅ Sigmoid 제거
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 데이터 로드
# -------------------------------
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # 입력(X), 타깃(y)
    X = df.select_dtypes(include=["number"]).drop(columns=["stress_label"], errors="ignore")
    y = df["stress_label"].astype(float)

    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)

    # train/test split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    return train_ds, test_ds, X.shape[1]

# -------------------------------
# 학습 루프
# -------------------------------
def train_model(model, train_ds, test_ds, epochs=30, lr=1e-3):
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    # ✅ 손실 함수 변경 (Sigmoid 제거했으므로 logits 전용)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # -------------------------------
    # 평가
    # -------------------------------
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # ✅ 평가 시 Sigmoid 적용
            y_pred = torch.sigmoid(model(X_batch))
            preds.extend((y_pred > 0.5).int().numpy().flatten())
            trues.extend(y_batch.numpy().flatten())

    acc = accuracy_score(trues, preds)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
    print(classification_report(trues, preds, target_names=["No Stress", "Stress"]))
    return model

# -------------------------------
# 메인 함수
# -------------------------------
def main():
    print("=== Stress Prediction Model Training Start ===")
    train_ds, test_ds, input_dim = load_dataset()
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}, Input dim: {input_dim}")

    model = StressModel(input_dim)
    trained_model = train_model(model, train_ds, test_ds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# -------------------------------
if __name__ == "__main__":
    main()
