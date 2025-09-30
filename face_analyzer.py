# face_analyzer.py (Final, Corrected Version)

import cv2
import numpy as np
import onnxruntime as ort
import config
from ultralytics import YOLO

class FaceAnalysis:
    def __init__(self):
        # 使用Ultralytics加載您的主模型
        print("正在使用Ultralytics加載您的自定義全能模型...")
        self.primary_detector = YOLO(config.PRIMARY_DETECTOR_PATH)
        print("✅ 主模型加載成功！")

        # --- 核心修正：重新加載所有其他必要的ONNX Runtime模型 ---
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print("正在加載人臉檢測模型...")
        self.face_detector = ort.InferenceSession(config.FACE_DETECTOR_PATH, providers=providers)
        self.face_input_name = self.face_detector.get_inputs()[0].name

        print("正在加載人臉識別模型...")
        self.recognizer = ort.InferenceSession(config.FACE_RECOGNIZER_PATH, providers=providers)
        self.recognizer_input_name = self.recognizer.get_inputs()[0].name

        print("正在加載ReID模型...")
        self.reid_model = ort.InferenceSession(config.REID_MODEL_PATH, providers=providers)
        self.reid_input_name = self.reid_model.get_inputs()[0].name
        
    def analyze_frame(self, frame):
        # 使用Ultralytics進行主推理
        results = self.primary_detector.predict(frame, verbose=False, conf=config.PERSON_CONF_THRESHOLD)
        result = results[0]

        # 將檢測結果分類
        all_objects = []
        for box in result.boxes:
            all_objects.append({
                'box': [int(coord) for coord in box.xyxy[0]],
                'class_id': int(box.cls[0]),
                'score': float(box.conf[0])
            })

        persons = [obj for obj in all_objects if obj['class_id'] == config.INTEREST_CLASSES['person']]
        behavior_objects = [obj for obj in all_objects if obj['class_id'] != config.INTEREST_CLASSES['person']]

        analysis_results = []
        for person in persons:
            p_box = person['box']
            px1, py1, px2, py2 = p_box
            person_roi = frame[py1:py2, px1:px2]

            person_data = {"person_box": p_box, "faces": [], "actions": []}
            
            # (行為推斷和人臉檢測/識別的後續邏輯保持不變)
            detected_actions = self._infer_actions(p_box, behavior_objects)
            if detected_actions:
                person_data["actions"] = detected_actions
            
            # if person_roi.size > 0:
            #     face_boxes_relative = self._detect_face(person_roi) 
            #     for f_box in face_boxes_relative:
            #         fx1_rel, fy1_rel, fx2_rel, fy2_rel = f_box
            #         abs_fx1, abs_fy1 = px1 + fx1_rel, py1 + fy1_rel; abs_fx2, abs_fy2 = px1 + fx2_rel, py1 + fy2_rel
            #         face_roi = frame[abs_fy1:abs_fy2, abs_fx1:abs_fx2]
            #         if face_roi.size > 0:
            #             embedding = self.get_embedding(face_roi)
            #             person_data["faces"].append({"face_box": (abs_fx1, abs_fy1, abs_fx2, abs_fy2), "embedding": embedding})
            analysis_results.append(person_data)
        return analysis_results
    
    # (所有輔助函數 _infer_actions, _detect_face, get_embedding 等都無需修改)
    def _infer_actions(self, p_box, objects):
        actions = []; px1, py1, px2, py2 = p_box; person_upper_body_y = py1 + (py2 - py1) * 0.6
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj['box']; obj_center_y = oy1 + (oy2 - oy1) / 2
            if not (ox2 < px1 or ox1 > px2 or oy2 < py1 or oy1 > py2):
                if obj['class_id'] in config.SMOKING_RELATED_IDS and obj_center_y < person_upper_body_y:
                    actions.append({"name": "Smoking", "evidence_box": obj['box']})
                elif obj['class_id'] in config.DRINKING_RELATED_IDS and obj_center_y < person_upper_body_y:
                    actions.append({"name": "Drinking", "evidence_box": obj['box']})
                elif obj['class_id'] in config.EATING_RELATED_IDS:
                    actions.append({"name": "Eating", "evidence_box": obj['box']})
        return actions
        
    def _detect_face(self, image):
        img_height, img_width = image.shape[:2]; input_img = cv2.resize(image, config.DETECTOR_INPUT_SIZE); input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB); input_img = input_img / 255.0; input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        outputs = self.face_detector.run(None, {self.face_input_name: input_tensor})[0]
        predictions = np.squeeze(outputs).T; scores = predictions[:, 4]; predictions = predictions[scores > config.FACE_CONF_THRESHOLD, :]; scores = scores[scores > config.FACE_CONF_THRESHOLD]
        boxes = predictions[:, :4]; x_factor = img_width / config.DETECTOR_INPUT_SIZE[0]; y_factor = img_height / config.DETECTOR_INPUT_SIZE[1]
        center_x = boxes[:, 0] * x_factor; center_y = boxes[:, 1] * y_factor; width = boxes[:, 2] * x_factor; height = boxes[:, 3] * y_factor
        x1 = center_x - width / 2; y1 = center_y - height / 2; x2 = center_x + width / 2; y2 = center_y + height / 2
        scaled_boxes = np.column_stack((x1, y1, x2, y2))
        indices = cv2.dnn.NMSBoxes([b.tolist() for b in scaled_boxes], scores.tolist(), config.FACE_CONF_THRESHOLD, config.NMS_THRESHOLD)
        final_boxes = [];
        if len(indices) > 0:
            for i in indices.flatten(): final_boxes.append(scaled_boxes[i].astype(int))
        return final_boxes
        
    def get_embedding(self, face_image):
        face_blob = cv2.dnn.blobFromImage(face_image, 1.0 / 127.5, config.RECOGNIZER_INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True)
        embedding = self.recognizer.run(None, {self.recognizer_input_name: face_blob})[0]; embedding = embedding.flatten(); embedding /= np.linalg.norm(embedding)
        return embedding
        
    def get_appearance_embedding(self, person_roi):
        reid_blob = cv2.dnn.blobFromImage(person_roi, 1.0 / 255.0, config.REID_INPUT_SIZE, (0.485, 0.456, 0.406), swapRB=True); reid_blob /= np.asarray([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]
        embedding = self.reid_model.run(None, {self.reid_input_name: reid_blob})[0]; embedding = embedding.flatten(); embedding /= np.linalg.norm(embedding)
        return embedding
        
    @staticmethod
    def cosine_similarity(vec1, vec2): return np.dot(vec1, vec2)