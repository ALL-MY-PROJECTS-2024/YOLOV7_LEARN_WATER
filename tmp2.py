import cv2
from flask_cors import CORS
import time
from threading import Thread, Event
from queue import Queue
import cv2
from flask import Flask, Response, request, jsonify
from threading import Thread, Event
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
import torch.backends.cudnn as cudnn
import torch
from pathlib import Path
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
import numpy as np  # [추가]
from uuid import uuid4  # [추가] 고유 ID 생성을 위해 사용

import threading
from utils.datasets import letterbox
from utils.plots import Annotator, colors, save_one_box

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


    # YOLOv7 설정
    conf_thres = 0.1
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
    line_thickness = 2
    device = select_device("cpu")
    model = DetectMultiBackend("best.pt", device=device, dnn=False, data="data/water.yaml", fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((320, 320), s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))  # 모델 워밍업


   

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


            #---------------------------------------
            # YOLOV7 MASKING 
            #---------------------------------------

            # frame 데이터에 LoadImages와 동일한 전처리 적용
            img_size = 320
            stride = 32
            auto = True


            #Letterbox 적용 (padding 및 크기 조정)
            preprocessed_frame = letterbox(frame, img_size, stride=stride, auto=auto)[0]  # 크기 조정된 이미지 반환
            # HWC -> CHW, BGR -> RGB 변환
            preprocessed_frame = preprocessed_frame.transpose((2, 0, 1))[::-1]
            preprocessed_frame = np.ascontiguousarray(preprocessed_frame)  # 메모리 연속성 보장

            # NumPy 배열 -> PyTorch 텐서
            im0 = torch.from_numpy(preprocessed_frame).to(device)
            im0 = im0.half() if model.fp16 else im0.float()  # 데이터 타입 설정
            im0 /= 255.0  # 정규화
            im0 = im0[None]  # 배치 차원 추가
            
            



            pred, out = model(im0, augment=False, visualize=False)
            proto = out[1]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            
           
            

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Mask plotting ----------------------------------------------------------------------------------------
            #mcolors = [colors(int(cls), True) for cls in det[:, 5]]
            # im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
            # annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
            # Mask plotting ----------------------------------------------------------------------------------------
            for i, det in enumerate(pred):  # 이미지당 처리
                if len(det):  # 감지된 객체가 있을 경우


                    # 원본 이미지로 좌표 스케일 조정
                    det[:, :4] = scale_coords(im0.shape[2:], det[:, :4], frame.shape).round()
                    # 마스크 처리    
                    # Process
                    im = letterbox(im0, img_size, stride=32)[0]  # resize
                    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    im = np.ascontiguousarray(im)  # contiguous
                    
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                    
                    
                    for idx, mask in enumerate(masks):
                        print(f"Filtered Mask {idx} unique values:", np.unique(mask.cpu().numpy()))


                    # # 디버깅: 각 마스크 값 확인
                    if masks is None or len(masks) == 0:
                        print("No masks generated.")
                    else:
                        print(f"Processed mask shape: {masks.shape}")
                        for idx, mask in enumerate(masks):
                            print(f"Mask {idx} unique values: {np.unique(mask.cpu().numpy())}")
                    #-----------------------------
                    # 라벨링
                    #-----------------------------
                    # 경계 상자 및 마스크 시각화
                    annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                    for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                        c = int(cls)  # 클래스 ID
                        label = f"{names[c]} {conf:.2f}"  # 라벨 생성
                        # print("label",label)

                        # 경계 상자 그리기
                        annotator.box_label(xyxy, label, color=colors(c, True))



                    # #     # 마스크 시각화
                    # mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                   
                    # # # Write results
                    # # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    # # im_masks = plot_masks(frame_tensor, masks, mcolors)  # 마스크 추가
                    # #frame = scale_masks(im0.shape[2:], im_masks, frame.shape)  # 마스크 크기 조정


                    # #-----------------------------
                    # # 마스킹
                    # #-----------------------------
                    # # 마스크 크기 조정: 모델 출력 크기 -> 원본 이미지 크기
                    # mask_size = (frame.shape[1], frame.shape[0])  # (width, height)
                    # masks_resized = [cv2.resize(mask.cpu().numpy(), mask_size) for mask in masks]

                    # # 마스크와 색상 오버레이
                    # for mask, color in zip(masks_resized, mcolors):
                    #     # 바이너리 마스크 변환
                    #     mask_binary = (mask > 0.3).astype(np.uint8)  # 0.5 대신 0.3 등으로 조정 가능

                    #     # 디버깅 출력
                    #     print("Mask unique values:", np.unique(mask_binary))
                        
                    #     # 색상 마스크 생성
                    #     color_mask = np.zeros_like(frame, dtype=np.uint8)
                    #     color_mask[:, :, 0] = mask_binary * int(color[0] * 255)  # Blue
                    #     color_mask[:, :, 1] = mask_binary * int(color[1] * 255)  # Green
                    #     color_mask[:, :, 2] = mask_binary * int(color[2] * 255)  # Red

                    #     # 마스크 오버레이 추가
                    #     frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)


                    # #-----------------------------
                    # # 윤곽선 그리기
                    # #-----------------------------
  


                    # # 최종 결과
                    # frame = annotator.result()

                else:
                    print("No objects detected.")

            #---------------------------------------
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
