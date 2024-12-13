import torch
import numpy as np
import cv2
from flask import Flask, Response, request, jsonify
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading
from flask_cors import CORS
import yaml  # YAML 파일을 읽기 위한 라이브러리
import random

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용

# YOLOv7 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 가중치 및 데이터셋 YAML 파일 설정
weights = 'runs/train/water_haed_finetuned/weights/best.pt'  # 적절한 경로로 수정
yaml_file = "./data/water.yaml"  # 클래스 정의가 포함된 YAML 파일

# YOLOv7 모델 직접 로드
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# YAML 파일에서 클래스 이름 로드
with open(yaml_file, "r", encoding="utf-8") as file:  # UTF-8 인코딩 명시
    data_yaml = yaml.safe_load(file)

# 클래스 이름 가져오기
model_classes = data_yaml["names"]  # YAML에서 클래스 이름 가져오기
print("클래스 이름:", model_classes)

# 클래스별 고유 색상 생성
colors = {cls: [random.randint(0, 255) for _ in range(3)] for cls in model_classes}

# 모델 로드 및 클래스 적용
model = attempt_load(weights, map_location=device)
model.names = model_classes  # 모델에 클래스 이름 적용
model.eval()

executor = ThreadPoolExecutor(max_workers=8)  # 워커 수 증가

# 스트림 프로세스를 관리하기 위한 딕셔너리와 락
stream_processes = {}
process_lock = threading.Lock()


def frame_to_jpeg(frame):
    """
    numpy 배열을 JPEG 이미지로 변환.
    """
    _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return jpeg.tobytes()


def detect_objects(frame):
    """
    YOLOv7 모델로 객체 탐지 수행.
    """
    resized_frame = cv2.resize(frame, (320, 320))  # YOLO 입력 크기를 320x320으로 설정
    img = torch.from_numpy(resized_frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    return pred


def plot_one_box_mod(box, img, label=None, color=(255, 0, 0), line_thickness=2):
    """
    객체 탐지를 표시하는 박스를 그리는 함수.
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_thickness)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 2), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)


def process_stream(m3u8_url):
    """
    FFmpeg로 스트림을 읽고 YOLOv7으로 객체 탐지 후 결과를 반환하는 제너레이터.
    """
    ffmpeg_command = [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", m3u8_url,
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-s", "320x320",  # 해상도를 320x320으로 설정
        "-r", "15",
        "-vf", "fps=10",
        "-an", "-",
    ]

    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    with process_lock:
        stream_processes[m3u8_url] = process

    def read_frame():
        raw_frame = process.stdout.read(320 * 320 * 3)  # 해상도에 맞게 읽기
        if not raw_frame:
            return None
        frame = np.frombuffer(raw_frame, np.uint8).reshape((320, 320, 3))
        return frame

    try:
        while True:
            frame = executor.submit(read_frame).result()
            if frame is None:
                print("Stream ended.")
                break

            # NumPy 배열 복사본 생성 (수정 가능 상태로 변경)
            frame = frame.copy()

            # YOLO 객체 탐지
            detections = detect_objects(frame)

            # 탐지 결과를 프레임에 시각화
            if detections:
                for det in detections:
                    if len(det):
                        det[:, :4] = scale_coords((320, 320), det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            cls_name = model.names[int(cls)]
                            color = colors[cls_name]  # 클래스별 색상 선택
                            label = f"{cls_name} {conf:.2f}"  # 클래스명 및 신뢰도
                            plot_one_box_mod(xyxy, frame, label=label, color=color, line_thickness=2)

            # JPEG 변환
            jpeg_frame = frame_to_jpeg(frame)

            # 스트리밍 반환
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg_frame + b"\r\n")
    finally:
        print("Terminating FFmpeg process.")
        with process_lock:
            if m3u8_url in stream_processes:
                del stream_processes[m3u8_url]
        process.terminate()
        process.wait()


@app.route("/stream")
def stream():
    """
    객체 탐지된 스트리밍을 클라이언트에 반환.
    """
    m3u8_url = request.args.get("hlsAddr")
    if not m3u8_url:
        return "hlsAddr 파라미터가 필요합니다.", 400

    print(f"Starting stream for: {m3u8_url}")
    return Response(process_stream(m3u8_url),
                    content_type="multipart/x-mixed-replace; boundary=frame",
                    status=200)


@app.route("/stop-stream", methods=["POST"])
def stop_stream():
    """
    특정 스트림을 종료.
    """
    data = request.json
    if not data or "hlsAddr" not in data:
        return jsonify({"error": "hlsAddr 파라미터가 필요합니다."}), 400

    m3u8_url = data["hlsAddr"]
    with process_lock:
        if m3u8_url in stream_processes:
            process = stream_processes[m3u8_url]
            process.terminate()
            process.wait()
            del stream_processes[m3u8_url]
            print(f"Stream for {m3u8_url} stopped successfully.")
            return jsonify({"message": f"Stream for {m3u8_url} stopped successfully."}), 200
        else:
            return jsonify({"error": f"No active stream found for {m3u8_url}."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
