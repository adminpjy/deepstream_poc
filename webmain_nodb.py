# --- 1. 导入和日志设置 ---
import cv2
import time
import numpy as np
import config
import threading
import json
from flask import Flask, render_template, Response, request, jsonify
import logging
from collections import deque

# ONNX Runtime 相关导入
import onnxruntime as ort

# --- 日誌設定 ---
log_queue = deque(maxlen=200)

class QueueLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.append(log_entry)

# 設定根日誌記錄器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

queue_handler = QueueLogHandler()
logging.getLogger().addHandler(queue_handler)

# 禁用Flask的默認INFO日誌
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 2. VideoAnalyzer 类 ---
class VideoAnalyzer:
    def __init__(self):
        # 初始化 ONNX 模型
        self.detector = ort.InferenceSession(config.PRIMARY_DETECTOR_PATH)
        
        self.cap = None
        self.running = False
        self.current_frame = None
        self.analysis_results = []
        self.lock = threading.Lock()

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                logging.error("错误：无法打开摄像头。")
                return False
            self.running = True
            self.thread = threading.Thread(target=self._analyze_loop)
            self.thread.start()
            logging.info("摄像头线程已启动。")
            return True
        return True

    def stop_camera(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.info("摄像头线程已停止。")

    def _analyze_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("无法从摄像头读取帧。")
                time.sleep(1)
                continue
            
            with self.lock:
                # 使用 ONNX 模型进行推理
                self.analysis_results = self._analyze_frame(frame)
                
                # 绘制检测结果
                for person_data in self.analysis_results:
                    px1, py1, px2, py2 = person_data["person_box"]
                    person_color = (255, 0, 0)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), person_color, 2)

                    actions = person_data.get("actions", [])
                    if actions:
                        action_texts = [action.get("name") for action in actions if action.get("name")]
                        for action in actions:
                            if "evidence_box" in action:
                                ex1, ey1, ex2, ey2 = action["evidence_box"]
                                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 165, 255), 2)
                        if action_texts:
                            cv2.putText(frame, ", ".join(action_texts), (px1, py2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                self.current_frame = frame
            time.sleep(0.01)

    def _analyze_frame(self, frame):
        # 预处理图像
        input_shape = config.DETECTOR_INPUT_SIZE
        original_height, original_width = frame.shape[:2]
        
        # 调整图像大小
        resized_frame = cv2.resize(frame, input_shape)
        # 转换为模型输入格式
        input_tensor = resized_frame.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # 执行推理
        outputs = self.detector.run(None, {'images': input_tensor})[0]
        
        # 后处理检测结果
        results = []
        for detection in outputs[0]:
            confidence = detection[4]
            if confidence > config.PERSON_CONF_THRESHOLD:
                class_id = int(np.argmax(detection[5:]))
                
                # 计算边界框坐标（相对于原始图像）
                x_center = detection[0] * original_width / input_shape[0]
                y_center = detection[1] * original_height / input_shape[1]
                width = detection[2] * original_width / input_shape[0]
                height = detection[3] * original_height / input_shape[1]
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # 根据类别处理检测结果
                if class_id == config.INTEREST_CLASSES["person"]:
                    # 为人员创建条目
                    results.append({
                        "person_box": [x1, y1, x2, y2],
                        "actions": [],
                        "faces": []
                    })
                else:
                    # 为行为创建条目
                    action_name = config.INTEREST_CLASSES_NAMES.get(class_id, f"Class_{class_id}")
                    if results:  # 如果已有人员检测结果，将行为添加到最近的人员
                        results[-1]["actions"].append({
                            "name": action_name.capitalize(),
                            "evidence_box": [x1, y1, x2, y2]
                        })
                    else:  # 如果没有人员检测结果，创建一个虚拟的人员条目
                        results.append({
                            "person_box": [x1, y1, x2, y2],
                            "actions": [{
                                "name": action_name.capitalize(),
                                "evidence_box": [x1, y1, x2, y2]
                            }],
                            "faces": []
                        })
        
        return results

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if ret: 
                    return buffer.tobytes()
            return None

# --- 3. Flask 应用程式設定 ---
analyzer = VideoAnalyzer()
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = analyzer.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_stream')
def log_stream():
    def generate_logs():
        while True:
            if log_queue:
                log_entry = log_queue.popleft()
                yield f"data: {log_entry}\n\n"
            time.sleep(0.1)
    return Response(generate_logs(), mimetype='text/event-stream')

@app.route('/start')
def start():
    if analyzer.start_camera(): 
        return jsonify({'success': True, 'message': '摄像头已启动'})
    return jsonify({'success': False, 'message': '无法启动摄像头'})

@app.route('/stop')
def stop():
    analyzer.stop_camera()
    return jsonify({'success': True, 'message': '摄像头已停止'})

def main():
    logging.info("启动智能分析系统 Web 服务...")
    logging.info("请在浏览器中访问 http://<your_ip>:5000")
    analyzer.start_camera()
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()