# file:/workspace/webmain.py (with logging optimization)
import cv2
import time
import numpy as np
import config
from face_analyzer import FaceAnalysis
from database_manager import DatabaseManager
import threading
import json
from flask import Flask, render_template, Response, request, jsonify
import logging
from collections import deque

# --- 1. 日誌設定 ---
log_queue = deque(maxlen=200)

class QueueLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.append(log_entry)

# 設定根日誌記錄器，捕獲我們應用程式的所有日誌
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

queue_handler = QueueLogHandler()
logging.getLogger().addHandler(queue_handler)

# --- 核心修改：獲取並禁用Flask的默認INFO日誌 ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# --- 2. VideoAnalyzer 類 (與之前版本完全相同) ---
class VideoAnalyzer:
    def __init__(self):
        self.db_manager = DatabaseManager(config.DB_NAME, config.DB_USER, config.DB_PASSWORD, config.DB_HOST, config.DB_PORT)
        self.face_analyzer = FaceAnalysis()
        self.face_database_centroids = self.db_manager.load_database_into_memory()
        self.cap = None
        self.running = False
        self.current_frame = None
        self.analysis_results = []
        self.recently_seen_unknown_faces = []
        self.UNKNOWN_FACE_COOLDOWN = 10
        self.pending_registration = None
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
                self.analysis_results = self.face_analyzer.analyze_frame(frame)
                
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

                    # if person_data["faces"]:
                    #     face_data = person_data["faces"][0]
                    #     fx1, fy1, fx2, fy2 = face_data["face_box"]
                    #     current_vector = face_data["embedding"]
                        
                    #     best_match_name, best_similarity = "Unknown", 0.0
                        
                    #     if self.face_database_centroids:
                    #         for name, centroid_vec in self.face_database_centroids.items():
                    #             similarity = self.face_analyzer.cosine_similarity(centroid_vec, current_vector)
                    #             if similarity > best_similarity:
                    #                 best_similarity, best_match_name = similarity, name
                        
                    #     if best_similarity > config.SIMILARITY_THRESHOLD:
                    #         label = f"{best_match_name} ({best_similarity:.2f})"
                    #         color = (0, 255, 0)
                    #     else:
                    #         label = f"Unknown ({best_similarity:.2f})"
                    #         color = (0, 0, 255)
                            
                    #         is_new_unknown = True
                    #         for seen_face in self.recently_seen_unknown_faces:
                    #             if self.face_analyzer.cosine_similarity(current_vector, seen_face['vector']) > 0.8:
                    #                 is_new_unknown = False
                    #                 if time.time() - seen_face['time'] > self.UNKNOWN_FACE_COOLDOWN:
                    #                     seen_face['time'] = time.time()
                    #                     is_new_unknown = True
                    #                 break
                            
                    #         if is_new_unknown and self.pending_registration is None:
                    #             self.recently_seen_unknown_faces.append({'vector': current_vector, 'time': time.time()})
                    #             self.pending_registration = {'vector': current_vector, 'similarity': best_similarity, 'timestamp': time.time()}
                    #             logging.warning(f"发现未知人员！(相似度: {best_similarity:.2f})")

                    #     cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                    #     cv2.putText(frame, label, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                current_time = time.time()
                self.recently_seen_unknown_faces = [f for f in self.recently_seen_unknown_faces if current_time - f['time'] < self.UNKNOWN_FACE_COOLDOWN]
                
                if self.pending_registration and current_time - self.pending_registration['timestamp'] > 30:
                    self.pending_registration = None
                
                self.current_frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if ret: return buffer.tobytes()
            return None

    def get_pending_registration(self):
        with self.lock: return self.pending_registration

    def register_person(self, person_name, employee_id):
        with self.lock:
            if self.pending_registration and person_name and employee_id:
                self.db_manager.add_embedding(person_name, employee_id, self.pending_registration['vector'])
                self.face_database_centroids = self.db_manager.load_database_into_memory()
                self.pending_registration = None
                logging.info(f"新人员 [{person_name}] 注册成功！")
                return True
            logging.error("注册失败：待注册信息无效。")
            return False

# --- 3. Flask 應用程式設定 (與之前版本完全相同) ---
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

@app.route('/status')
def status():
    pending = analyzer.get_pending_registration()
    if pending:
        similarity_score = float(pending['similarity'])
        return jsonify({'pending_registration': True, 'similarity': similarity_score})
    return jsonify({'pending_registration': False})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if analyzer.register_person(data.get('name'), data.get('employee_id')):
        return jsonify({'success': True, 'message': '注册成功！'})
    return jsonify({'success': False, 'message': '注册失败！'})

@app.route('/start')
def start():
    if analyzer.start_camera(): return jsonify({'success': True, 'message': '摄像头已启动'})
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