import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# 영상 저장 관련 설정
SAVE_PATH = "output.avi"  # 저장할 영상 파일 경로
FPS = 20.0  # 저장할 프레임 레이트
FRAME_SIZE = (640, 480)  # 프레임 크기 (width, height)

def stream_video(source):
    cap = cv2.VideoCapture(source)

    # 저장을 위한 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(SAVE_PATH, fourcc, FPS, FRAME_SIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Ending stream.")
                break

            # 프레임 리사이즈 (저장 용도)
            resized_frame = cv2.resize(frame, FRAME_SIZE)

            # 프레임 저장
            out.write(resized_frame)

            # 프레임 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_bytes = buffer.tobytes()

            # Flask Response로 프레임 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        out.release()  # 저장 종료
        print(f"Video saved to {SAVE_PATH}")

@app.route('/stream', methods=['GET'])
def video_feed():
    # 비디오 스트림 링크 설정
    source = "https://safecity.busan.go.kr/playlist/cnRzcDovL2d1ZXN0Omd1ZXN0QDEwLjEuMjEwLjIxNjo1NTQvdXM2NzZyM0RMY0RuczYwdE5EQXk=/index.m3u8"  # 스트리밍 링크를 입력하세요
    return Response(stream_video(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop-save', methods=['POST'])
def stop_save():
    """저장 중지 기능: 추가적으로 구현할 수 있음"""
    return jsonify({"message": "Stop save functionality can be implemented here."})

if __name__ == '__main__':
    # Flask 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=True)
