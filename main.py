import cv2
import time
import numpy as np
import config
from face_analyzer import FaceAnalysis
from database_manager import DatabaseManager

def main():
    # --- 初始化 ---
    db_manager = DatabaseManager(config.DB_NAME, config.DB_USER, config.DB_PASSWORD, config.DB_HOST, config.DB_PORT)
    face_analyzer = FaceAnalysis()
    face_database_centroids = db_manager.load_database_into_memory()
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("错误：无法打开摄像头。"); return

    print("启动智能分析系统 (带自动注册功能)...")
    print("按 'q' 键退出。")

    # 用于防止对同一个未知人员反复提示注册
    recently_seen_unknown_faces = []
    UNKNOWN_FACE_COOLDOWN = 10 # 秒

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- 核心流程：持续分析 ---
        analysis_results = face_analyzer.analyze_frame(frame)
        
        for person_data in analysis_results:
            px1, py1, px2, py2 = person_data["person_box"]
            
            # 默认标签和颜色
            person_label = "Person"
            person_color = (255, 0, 0) # 蓝色

            # 绘制人形框
            cv2.rectangle(frame, (px1, py1), (px2, py2), person_color, 2)

            # --- 行为分析与绘制 ---
            actions = person_data.get("actions", [])
            if actions:
                action_texts = []
                for action in actions:
                    action_name = action.get("name")
                    if action_name: action_texts.append(action_name)
                    if "evidence_box" in action:
                        ex1, ey1, ex2, ey2 = action["evidence_box"]
                        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 165, 255), 2)
                
                if action_texts:
                    action_summary = ", ".join(action_texts)
                    cv2.putText(frame, action_summary, (px1, py2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # --- 人脸识别与自动注册 ---
            if person_data["faces"]:
                face_data = person_data["faces"][0]
                fx1, fy1, fx2, fy2 = face_data["face_box"]
                current_vector = face_data["embedding"]
                
                best_match_name = "Unknown"
                best_similarity = 0.0
                
                if face_database_centroids:
                    for name, centroid_vec in face_database_centroids.items():
                        similarity = face_analyzer.cosine_similarity(centroid_vec, current_vector)
                        if similarity > best_similarity:
                            best_similarity, best_match_name = similarity, name
                
                if best_similarity > config.SIMILARITY_THRESHOLD:
                    # 识别成功
                    label = f"{best_match_name} ({best_similarity:.2f})"
                    color = (0, 255, 0) # 绿色
                else:
                    # 未知人员
                    label = f"Unknown ({best_similarity:.2f})"
                    color = (0, 0, 255) # 红色

                    # --- 自动注册逻辑 ---
                    # 检查是否是最近已经提示过的未知人员
                    is_new_unknown = True
                    for seen_face in recently_seen_unknown_faces:
                        sim_to_seen = face_analyzer.cosine_similarity(current_vector, seen_face['vector'])
                        if sim_to_seen > 0.8: # 如果和某个已提示的陌生人很像，就认为是同一个人
                            is_new_unknown = False
                            # 检查冷却时间是否已过
                            if time.time() - seen_face['time'] > UNKNOWN_FACE_COOLDOWN:
                                seen_face['time'] = time.time() # 更新时间，重新开始冷却
                                is_new_unknown = True
                            break
                    
                    if is_new_unknown:
                        recently_seen_unknown_faces.append({'vector': current_vector, 'time': time.time()})
                        
                        print("\n" + "="*30)
                        print(f"⚠️  发现未知人员！(相似度: {best_similarity:.2f})")
                        choice = input(">> 是否要为他/她注册? (y/n): ").lower()
                        
                        if choice == 'y':
                            person_name = input(">> 请输入姓名: ")
                            employee_id = input(">> 请输入工号: ")
                            if person_name and employee_id:
                                # 使用当前高质量的帧进行注册
                                db_manager.add_embedding(person_name, employee_id, current_vector)
                                print("✅ 注册成功！正在更新识别数据库...")
                                face_database_centroids = db_manager.load_database_into_memory()
                                label = person_name # 立即更新标签
                                color = (0, 255, 0)
                            else:
                                print("取消注册。")
                        else:
                            print("已选择不注册。")
                        print("="*30 + "\n")

                # 绘制人脸框和标签
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                cv2.putText(frame, label, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 清理过时的未知人员记录
        recently_seen_unknown_faces = [f for f in recently_seen_unknown_faces if time.time() - f['time'] < UNKNOWN_FACE_COOLDOWN]

        cv2.imshow("AI Analyzer with Auto-Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()