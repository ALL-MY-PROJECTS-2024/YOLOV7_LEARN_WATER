import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D


# ===========================
# 1. 데이터 준비
# ===========================
def load_video_frames(video_path, frame_count=20, resize=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frame_count * 3:  # 3배로 읽고 샘플링
        ret, frame = cap.read()
        if not ret:
            print("Warning: No more frames to read. Check the video path or file.")
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), resize)  # 흑백 변환 및 리사이즈
        frames.append(frame / 255.0)  # 정규화
    cap.release()

    frames = frames[::3]  # 3프레임마다 샘플링
    print(f"Loaded {len(frames)} frames from {video_path} with resolution {resize}.")
    return np.array(frames)


# ===========================
# 2. ConvLSTM 모델 정의
# ===========================
def create_conv_lstm_model(input_shape):
    model = Sequential([
        # 첫 번째 ConvLSTM 레이어
        ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu',
                   input_shape=input_shape, padding='same', return_sequences=False),
        BatchNormalization(),

        # 최종 Conv2D 출력 레이어
        Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mae')  # mae 사용 (빠른 수렴)
    return model

# ===========================
# 3. 학습 및 예측
# ===========================
def train_and_predict(video_frames, prediction_steps=10):
    # 시계열 데이터 구성
    input_frames = []
    target_frames = []
    for i in range(len(video_frames) - prediction_steps):
        input_frames.append(video_frames[i:i + prediction_steps])  # 시퀀스 입력
        target_frames.append(video_frames[i + prediction_steps])   # 최종 예측 프레임

    X = np.array(input_frames).reshape(-1, prediction_steps, video_frames.shape[1], video_frames.shape[2], 1)
    y = np.array(target_frames).reshape(-1, video_frames.shape[1], video_frames.shape[2], 1)

    # 모델 생성 및 학습
    model = create_conv_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=10, batch_size=4, verbose=1)

    # 예측
    predicted_image = model.predict(X[-1].reshape(1, *X.shape[1:]))
    predicted_image = predicted_image[0, :, :, 0] * 255.0
    predicted_image = np.clip(predicted_image, 0, 255).astype(np.uint8)

    # 결과 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Input Frame")
    plt.imshow(video_frames[-1], cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Image")
    plt.imshow(predicted_image, cmap='gray')

    plt.show()


# ===========================
# 실행
# ===========================
if __name__ == "__main__":
    video_path = "output.avi"  # 영상 파일 경로
    video_frames = load_video_frames(video_path, frame_count=10, resize=(128, 128))  # 해상도 줄임
    train_and_predict(video_frames, prediction_steps=5)