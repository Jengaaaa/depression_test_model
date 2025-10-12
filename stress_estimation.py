import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# --------------------------
# 1. PyTorch 모델 정의 (간단한 MLP)
# --------------------------
class StressPredictor(nn.Module):
    def __init__(self, input_dim=478*2, hidden_dim=256):
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
# 2. Mediapipe 초기화
# --------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# --------------------------
# 3. 모델 로드 (초기엔 랜덤 가중치)
# --------------------------
model = StressPredictor()
model.eval()

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

    # 좌우반전 (자연스럽게)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # 얼굴 랜드마크 468개 (x, y)
            coords = []
            h, w, _ = frame.shape
            for lm in face_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

            # 예측 (스트레스 지수 0~1)
            with torch.no_grad():
                stress_score = model(coords).item()

            # 시각화
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

            cv2.putText(frame, f"Stress Score: {stress_score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Stress Estimation (Press Q to Exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
