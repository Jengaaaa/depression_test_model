import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

from FaceMeshModule import FaceMeshDetector  # 오픈소스 모듈 import


# --------------------------
# 1. PyTorch 모델 정의 (간단한 MLP)
# --------------------------
class StressPredictor(nn.Module):
    def __init__(self, input_dim=468 * 2, hidden_dim=256):   
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


# --------------------------
# 2. 모델 로드 (초기엔 랜덤 가중치)
# --------------------------
model = StressPredictor()
model.eval()

# --------------------------
# 3. FaceMeshDetector 초기화
# --------------------------
detector = FaceMeshDetector()

# --------------------------
# 4. 웹캠 스트리밍 시작
# --------------------------
cap = cv2.VideoCapture(0)  # 0: 기본 카메라

transform = transforms.Compose([
    transforms.ToTensor()
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전 (자연스럽게 보기 위해)
    frame = cv2.flip(frame, 1)

    # 얼굴 메쉬 탐지 및 시각화
    frame = detector.findFaceMesh(frame, draw=True)
    lmList = detector.findPosition(frame, draw=False)  # draw=True 하면 landmark 점도 표시됨

    if len(lmList) != 0:
        # (id, x, y) 중 x, y만 추출
        coords = np.array(lmList)[:, 1:3].flatten()

        # torch tensor로 변환
        coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

        # 모델 예측 (스트레스 지수: 0~1)
        with torch.no_grad():
            stress_score = model(coords).item()

        # 화면에 표시
        cv2.putText(frame, f"Stress Score: {stress_score:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    # 결과 화면 출력
    cv2.imshow("Stress Estimation (Press Q to Exit)", frame)

    # 종료 키(Q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
