import cv2
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import time
from threading import Thread, Event
from queue import Queue

app = Flask(__name__)
CORS(app)

# 스트림 및 클라이언트 관리 데이터 구조
stream_threads = {}  # {source: (producer_thread, stop_event, frame_queue)}
client_connections = {}  # {source: {client_id: True}}

# 클라이언트 고유 ID 생성
from uuid import uuid4


def process_stream(source, stop_event, frame_queue):
    """스트림 연결, 읽기 및 프레임 생산"""
    cap = None
    attempts = 0
    fps = 15  # 낮은 FPS로 설정
    last_frame_time = time.time()

    try:
        while not stop_event.is_set():
            # 스트림 연결 시도
            if cap is None or not cap.isOpened():
                if attempts >= 5:
                    raise RuntimeError(f"Failed to connect to stream {source} after 5 attempts.")
                cap = cv2.VideoCapture(source)

                if cap.isOpened():
                    print(f"Stream connected successfully: {source}")
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # 해상도 설정 (320x320)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

                    attempts = 0
                else:
                    print(f"Failed to connect to stream {source}. Attempt {attempts + 1}/5")
                    attempts += 1
                    time.sleep(3)
                    continue

            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {source}. Reconnecting...")
                cap.release()
                cap = None
                continue

            # FPS 제한
            current_time = time.time()
            frame_interval = 1.0 / fps
            elapsed_time = current_time - last_frame_time
            if elapsed_time < frame_interval:
                time.sleep(frame_interval - elapsed_time)
            last_frame_time = time.time()

            # 프레임을 큐에 넣기 (큐 크기 제한)
            if frame_queue.qsize() < 10:
                # 품질 최적화: 프레임 크기 조정 및 압축
                resized_frame = cv2.resize(frame, (320, 320))  # 해상도 강제 설정
                frame_queue.put(resized_frame)

    except Exception as e:
        print(f"Error in process_stream for {source}: {e}")
    finally:
        if cap:
            cap.release()
        print(f"Stopped processing stream for {source}.")


@app.route('/stream', methods=['GET'])
def stream_video():
    """스트림 요청"""
    source = request.args.get('hlsAddr')
    if not source:
        return jsonify({"error": "Missing 'hlsAddr' query parameter"}), 400

    # 클라이언트 고유 ID 생성
    client_id = str(uuid4())

    # 스트림이 이미 실행 중인지 확인
    if source not in stream_threads:
        # 새 스트림 시작
        stop_event = Event()
        frame_queue = Queue()
        stream_threads[source] = (Thread(target=process_stream, args=(source, stop_event, frame_queue)),
                                  stop_event, frame_queue)

        producer_thread, stop_event, frame_queue = stream_threads[source]
        producer_thread.start()
        client_connections[source] = {}

        print(f"Stream from {source} started.")

    # 클라이언트 연결 추가
    client_connections[source][client_id] = True

    # 클라이언트에게 clientId 반환
    return jsonify({"clientId": client_id}), 200


@app.route('/stream-data', methods=['GET'])
def stream_data():
    """실제 스트림 데이터 전송"""
    source = request.args.get('hlsAddr')
    client_id = request.args.get('clientId')

    if not source or not client_id:
        return jsonify({"error": "Missing 'hlsAddr' or 'clientId' parameter"}), 400

    if source not in stream_threads or client_id not in client_connections[source]:
        return jsonify({"error": "Invalid 'hlsAddr' or 'clientId'"}), 400

    def generate_frames():
        frame_queue = stream_threads[source][2]
        while source in client_connections and client_id in client_connections[source]:
            if not frame_queue.empty():
                frame = frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])  # 품질 설정 (낮은 품질)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)

        # 클라이언트 연결 종료
        client_connections[source].pop(client_id, None)
        if not client_connections[source]:
            stop_existing_stream(source)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop-stream', methods=['POST'])
def stop_stream():
    """특정 클라이언트 연결 종료"""
    data = request.get_json()
    source = data.get('hlsAddr')
    client_id = data.get('clientId')

    if not source or not client_id:
        return jsonify({"error": "Missing 'hlsAddr' or 'clientId' parameter"}), 400

    if source in client_connections and client_id in client_connections[source]:
        client_connections[source].pop(client_id, None)
        if not client_connections[source]:
            stop_existing_stream(source)
        return jsonify({"message": f"Client {client_id} disconnected from {source}."}), 200
    else:
        return jsonify({"error": f"Client {client_id} not found for source {source}"}), 404


def stop_existing_stream(source):
    """기존 스트림 안전 종료"""
    print(f"Stopping existing stream for {source}...")
    if source in stream_threads:
        producer_thread, stop_event, _ = stream_threads[source]
        stop_event.set()
        producer_thread.join(timeout=5)
        del stream_threads[source]
        del client_connections[source]
        print(f"Existing stream for {source} successfully stopped.")
    else:
        print(f"No active stream found for {source}.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
