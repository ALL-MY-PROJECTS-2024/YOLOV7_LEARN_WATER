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
def load_video_frames(video_path, frame_count=100, resize=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Warning: No more frames to read. Check the video path or file.")
            break
        frame = cv2.resize(frame, resize)  # 이미지 리사이즈
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        frames.append(frame / 255.0)  # 정규화
    cap.release()

    if len(frames) == 0:
        raise ValueError("Error: No frames loaded. Check the video file path or content.")
    
    print(f"Loaded {len(frames)} frames from {video_path}.")
    return np.array(frames)


# ===========================
# 2. ConvLSTM 모델 정의
# ===========================
def create_conv_lstm_model(input_shape):
    model = Sequential([
        ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu',
                   input_shape=input_shape, padding='same', return_sequences=True),
        BatchNormalization(),
        
        ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu',
                   padding='same', return_sequences=False),  # 마지막 시퀀스만 출력
        BatchNormalization(),
        
        Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')  # 최종 예측 이미지 출력
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ===========================
# 3. 학습 및 예측
# ===========================
def train_and_predict(video_frames, prediction_steps=10):
    # 시계열 데이터 구성
    input_frames = []
    target_frames = []
    for i in range(len(video_frames) - prediction_steps):
        input_frames.append(video_frames[i:i+prediction_steps])  # 시퀀스 입력
        target_frames.append(video_frames[i+prediction_steps])   # 최종 예측 프레임

    X = np.array(input_frames).reshape(-1, prediction_steps, video_frames.shape[1], video_frames.shape[2], 1)
    y = np.array(target_frames).reshape(-1, video_frames.shape[1], video_frames.shape[2], 1)  # 최종 출력 이미지

    # 모델 생성 및 학습
    model = create_conv_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=10, batch_size=4, verbose=1)

    # 예측
    predicted_image = model.predict(X[-1].reshape(1, *X.shape[1:]))  # 가장 마지막 시퀀스를 기반으로 예측
    predicted_image = predicted_image[0, :, :, 0]  # 이미지 차원 조정

    # 결과 시각화
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input Frame")
    plt.imshow(video_frames[-1], cmap='gray')  # 마지막 입력 이미지 표시

    plt.subplot(1, 2, 2)
    plt.title("Predicted Image")
    plt.imshow(predicted_image, cmap='gray')  # 예측된 이미지 표시

    plt.show()


# ===========================
# 실행
# ===========================
if __name__ == "__main__":
    # 영상 파일 경로 입력
    video_path = "output.avi"  # 영상 파일 경로
    video_frames = load_video_frames(video_path, frame_count=200)
    train_and_predict(video_frames)
