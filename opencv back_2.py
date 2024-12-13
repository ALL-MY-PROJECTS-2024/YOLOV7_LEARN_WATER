import cv2
import torch
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import numpy as np
import time
from threading import Thread, Event
import queue
from models.experimental import attempt_load
from utils.torch_utils import select_device

app = Flask(__name__)
CORS(app)

# YOLOv7-Seg 모델 로드
weights = 'yolov7-seg.pt'  # 모델 가중치 경로
device = select_device('cpu')  # '0'으로 설정 시 GPU 사용
model = attempt_load(weights, device=device)  # device로 변경
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names  # 클래스 이름

stream_threads = {}
stream_events = {}
frame_queues = {}


def connect_to_stream(source, retry_interval=3):
    cap = None
    attempts = 0
    while attempts < 5:
        if cap:
            cap.release()
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"Connected to stream: {source}")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            print(f"Failed to connect to {source}. Retry {attempts + 1}/5")
        attempts += 1
        time.sleep(retry_interval)
    raise RuntimeError(f"Failed to connect to stream {source}")



def process_frame(frame):
    """YOLOv7-Seg 모델을 사용하여 프레임 처리"""
    if frame is None or frame.size == 0:
        print("Warning: Received an empty frame. Skipping processing...")
        return frame

    # 이미지 전처리
    img = cv2.resize(frame, (320, 320))[:, :, ::-1]  # BGR -> RGB 변환 및 크기 조정
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device) / 255.0
    img = img.unsqueeze(0) if len(img.shape) == 3 else img

    # 모델 추론
    with torch.no_grad():
        outputs = model(img, augment=False)

    # 마스크 데이터 추출
    masks = outputs[1]  # 출력의 두 번째 요소에서 마스크 추출
    if isinstance(masks, tuple):
        masks = masks[1]  # 두 번째 튜플의 요소 선택

    print(f"Debug: Masks shape: {masks.shape if isinstance(masks, torch.Tensor) else 'Unknown'}")

    if masks is None or not isinstance(masks, torch.Tensor):
        print("Warning: No valid masks returned by the model. Skipping frame...")
        return frame

    # 마스크를 처리하고 프레임에 적용
    return process_frame_with_mask(frame, masks)




def process_frame_with_mask(frame, masks):
    """
    YOLOv7-Seg 마스크 데이터를 OpenCV로 처리
    """
    if frame is None or frame.size == 0:
        print("Warning: Received an empty frame. Skipping processing...")
        return frame

    if masks is None or len(masks) == 0:
        print("Warning: No masks returned by the model. Skipping frame...")
        return frame

    # 원본 프레임 크기
    height, width = frame.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # 전체 마스크 초기화

    # 마스크 데이터 처리
    mask_tensor = masks[0]  # 첫 번째 배치 선택
    mask_array = mask_tensor.cpu().numpy()  # Numpy 변환
    print(f"Debug: Mask shape before processing: {mask_array.shape}")

    for idx in range(mask_array.shape[0]):  # 각 채널 반복
        mask = mask_array[idx]  # (80, 80) 마스크
        mask_resized = cv2.resize(mask, (width, height))  # 원본 크기로 리사이즈
        mask_binary = (mask_resized > 0.5).astype(np.uint8)  # 이진화
        combined_mask = np.maximum(combined_mask, mask_binary)  # 마스크 병합

    # 컬러 마스크 생성
    mask_colored = cv2.merge([combined_mask * 0, combined_mask * 255, combined_mask * 0])  # 초록색 마스크

    # 원본 프레임과 병합
    frame_with_mask = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)

    return frame_with_mask





def read_frames(source, stop_event):
    cap = connect_to_stream(source)
    frame_queue = frame_queues[source]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / fps

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {source}. Reconnecting...")
                cap.release()
                cap = connect_to_stream(source)
                continue
            if frame_queue.qsize() < 50:
                frame_queue.put(frame)
            time.sleep(frame_interval)
    finally:
        cap.release()


def generate_stream(source, stop_event):
    frame_queue = frame_queues[source]
    try:
        while not stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                processed_frame = process_frame(frame)  # YOLOv7-Seg 마스킹 적용
                _, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)
    finally:
        print(f"Stopped stream generation for {source}.")


@app.route('/stream', methods=['GET'])
def stream_video():
    source = request.args.get('hlsAddr')
    if not source:
        return jsonify({"error": "Missing 'hlsAddr' query parameter"}), 400

    if source in stream_threads:
        stop_existing_stream(source)

    frame_queues[source] = queue.Queue()
    stop_event = Event()
    stream_events[source] = stop_event
    read_thread = Thread(target=read_frames, args=(source, stop_event))
    stream_threads[source] = read_thread
    read_thread.start()
    print(f"Stream started for {source}")

    return Response(
        generate_stream(source, stop_event),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def stop_existing_stream(source):
    print(f"Stopping stream for {source}")
    if source in stream_events:
        stream_events[source].set()
        stream_threads[source].join()
        del stream_threads[source]
        del stream_events[source]
        del frame_queues[source]
        print(f"Stream stopped for {source}")


@app.route('/stop-stream', methods=['POST'])
def stop_stream():
    data = request.get_json()
    source = data.get('hlsAddr')
    if not source:
        return jsonify({"error": "Missing 'hlsAddr' parameter"}), 400
    stop_existing_stream(source)
    return jsonify({"message": f"Stream from {source} stopped."}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
