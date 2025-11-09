import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from collections import deque
import numpy as np
import time

# -------------------------------
# 1. 환경 설정
# -------------------------------
MODEL_PATH = "./model/emotion_modelv2.pth"  # Colab에서 학습한 EfficientNet 모델 경로
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -------------------------------
# 2. 모델 불러오기
# -------------------------------
print("EfficientNet 모델 로딩 중...")
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("모델 로딩 완료!")

# -------------------------------
# 3. 전처리 정의
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

# -------------------------------
# 4. 얼굴 인식 설정
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------------------
# 5. 실시간 감정 분석 (프레임 속도 제한 + 평균 안정화 + 3초 유지)
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
    exit()

print("실시간 감정 감지 시작! (종료: Q키)")

pred_buffer = deque(maxlen=5)
fps_limit = 2
prev_time = 0

# 최근 감정 결과를 일정 시간(3초) 유지하기 위한 변수
last_display_time = 0
display_duration = 3.0  # 초 단위 (3초)
last_emotion = None
last_confidence = 0.0
last_risk_score = 0.0
last_box = None  # (x, y, w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    current_time = time.time()

    # 프레임 속도 제한 (2 FPS)
    if current_time - prev_time < 1.0 / fps_limit:
        # 마지막 감정 결과가 유지 시간 이내라면 그대로 표시
        if last_emotion and (current_time - last_display_time) < display_duration and last_box is not None:
            x, y, w, h = last_box
            color = (0, 0, 255) if last_risk_score >= 0.6 else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{last_emotion} ({last_confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Depression Risk: {last_risk_score:.2f}", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("DepVidMood - 실시간 감정 인식 및 우울 감지", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                img_tensor = transform(face_img).unsqueeze(0)
            except Exception:
                continue

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_buffer.append(pred_idx)

            stable_pred = max(set(pred_buffer), key=pred_buffer.count)
            emotion = CLASSES[stable_pred]
            confidence = probs[0][pred_idx].item()

            depression_map = {
                'Sad': 0.9, 'Fear': 0.7, 'Angry': 0.6,
                'Neutral': 0.4, 'Disgust': 0.5,
                'Surprise': 0.2, 'Happy': 0.1
            }
            risk_score = depression_map.get(emotion, 0.5)

            # 감정 결과 갱신
            last_emotion = emotion
            last_confidence = confidence
            last_risk_score = risk_score
            last_display_time = current_time
            last_box = (x, y, w, h)

    # 감정 유지 시간 내라면 계속 표시
    if last_emotion and (current_time - last_display_time) < display_duration and last_box is not None:
        x, y, w, h = last_box
        color = (0, 0, 255) if last_risk_score >= 0.6 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{last_emotion} ({last_confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Depression Risk: {last_risk_score:.2f}", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("DepVidMood - 실시간 감정 인식 및 우울 감지", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
