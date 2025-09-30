# database_manager.py
import numpy as np
import psycopg2
import json
from pgvector.psycopg2 import register_vector

class DatabaseManager:
    # ... (__init__, add_embedding, load_database_into_memory 等函数不变) ...
    def __init__(self, db_name, user, password, host="localhost", port="5432"):
        try:
            self.conn = psycopg2.connect(dbname=db_name, user=user, password=password, host=host, port=port)
            self.cursor = self.conn.cursor()
            register_vector(self.conn)
            print("✅ 成功连接到 PostgreSQL 数据库。")
        except psycopg2.OperationalError as e:
            print(f"❌ 无法连接到 PostgreSQL 数据库: {e}"); exit()
    def add_embedding(self, person_name, employee_id, vector):
        # ... (代码不变) ...
        try:
            self.cursor.execute("INSERT INTO employees (name, employee_id) VALUES (%s, %s) ON CONFLICT (employee_id) DO NOTHING", (person_name, employee_id))
            self.cursor.execute("SELECT id FROM employees WHERE employee_id=%s", (employee_id,)); person_id_tuple = self.cursor.fetchone()
            if not person_id_tuple:
                self.cursor.execute("SELECT id FROM employees WHERE name=%s", (person_name,)); person_id_tuple = self.cursor.fetchone()
            if not person_id_tuple: print(f"错误：无法找到或创建人员 {person_name}"); return
            person_id = person_id_tuple[0]
            self.cursor.execute("INSERT INTO face_embeddings (employee_id, embedding) VALUES (%s, %s)", (person_id, vector.astype(np.float32)))
            self.conn.commit()
            print(f"成功将 [{person_name}] 的一条新特征存入数据库。")
        except Exception as e:
            print(f"数据库操作失败: {e}"); self.conn.rollback()
    def load_database_into_memory(self):
        # ... (鲁棒质心计算逻辑不变) ...
        temp_db = {}
        try:
            self.cursor.execute("SELECT p.name, e.embedding FROM employees p JOIN face_embeddings e ON p.id = e.employee_id")
            rows = self.cursor.fetchall()
            for name, vector in rows:
                if name not in temp_db: temp_db[name] = []
                temp_db[name].append(vector)
        except Exception as e:
            print(f"从数据库加载特征失败: {e}"); return {}
        face_database_centroids = {}
        for name, vectors_list in temp_db.items():
            if not vectors_list: continue
            if len(vectors_list) < 4:
                centroid_vector = np.mean(vectors_list, axis=0)
            else:
                initial_centroid = np.mean(vectors_list, axis=0)
                distances = [1 - np.dot(initial_centroid, v) for v in vectors_list]
                vector_distance_pairs = sorted(zip(vectors_list, distances), key=lambda x: x[1])
                num_to_keep = int(len(vector_distance_pairs) * 0.75)
                good_vectors = [pair[0] for pair in vector_distance_pairs[:num_to_keep]]
                centroid_vector = np.mean(good_vectors, axis=0)
                print(f"对于 [{name}], 从 {len(vectors_list)} 个样本中剔除了 {len(vectors_list) - len(good_vectors)} 个离群点来计算质心。")
            centroid_vector /= np.linalg.norm(centroid_vector)
            face_database_centroids[name] = centroid_vector
        if face_database_centroids: print(f"成功从数据库加载并为 {len(face_database_centroids)} 人计算了鲁棒特征质心。")
        else: print("数据库为空，等待新的人脸注册。")
        return face_database_centroids

    # --- 新增函数 ---
    def get_employee_by_name(self, name):
        """根据姓名查询员工信息"""
        self.cursor.execute("SELECT id, employee_id, name, gender FROM employees WHERE name=%s", (name,))
        row = self.cursor.fetchone()
        if row:
            return {"id": row[0], "employee_id": row[1], "name": row[2], "gender": row[3]}
        return None

    def log_event(self, employee_id, camera_id, event_type, track_id):
        """记录出入事件"""
        self.cursor.execute("INSERT INTO access_log (employee_id, camera_id, event_type, track_id) VALUES (%s, %s, %s, %s)",
                            (employee_id, camera_id, event_type, track_id))
        self.conn.commit()

    def log_alarm(self, camera_id, event_type, details):
        """记录报警事件"""
        self.cursor.execute("INSERT INTO alarms (camera_id, event_type, details) VALUES (%s, %s, %s)",
                            (camera_id, event_type, json.dumps(details)))
        self.conn.commit()

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()