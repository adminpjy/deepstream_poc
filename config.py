# config.py (Final Version for Custom "all-in-one" Model)

# --- 1. 数据库配置 ---
DB_NAME = "postgresvision_db"
DB_USER = "admin"
DB_PASSWORD = "Password01!" 
DB_HOST = "localhost"
DB_PORT = "5432"

# --- 2. 模型路径配置 ---
# 核心修改：现在我们只有一个主检测器，就是您自己训练的全能模型
PERSON_MODEL = 'models/yolo11n.pt'
HAT_MODEL = 'models/hat.onnx'
PHONE_MODEL = 'models/phone.pt'
SMOKING_MODEL = 'models/smoking.pt'

PRIMARY_DETECTOR_PATH = 'models/best.pt' # <<-- 已更新为您的全能模型
FACE_DETECTOR_PATH = 'models/yolov11n-face.onnx'      
FACE_RECOGNIZER_PATH = 'models/w600k_r50.onnx'      
REID_MODEL_PATH = 'models/osnet_x0_25_msmt17_dynamic.onnx' 

# --- 3. 模型输入尺寸 ---
DETECTOR_INPUT_SIZE = (640, 640)
RECOGNIZER_INPUT_SIZE = (112, 112)
REID_INPUT_SIZE = (128, 256)

# --- 4. AI 阈值配置 ---
PERSON_CONF_THRESHOLD = 0.45    # 对所有物体的通用置信度阈值
FACE_CONF_THRESHOLD = 0.5      
NMS_THRESHOLD = 0.5           
SIMILARITY_THRESHOLD = 0.5    
REID_THRESHOLD = 0.5          

# --- 5. 行为分析配置 ---
# 核心修改：更新为最终的4个类别
INTEREST_CLASSES = {
    "person": 0,
    "smoking": 1,
    "drinking": 2,
    "eating": 3
}
# 新增一個反向映射，用於日誌打印
INTEREST_CLASSES_NAMES = {v: k for k, v in INTEREST_CLASSES.items()} 


# 为行为推断创建ID集合
SMOKING_RELATED_IDS = {INTEREST_CLASSES["smoking"]}
DRINKING_RELATED_IDS = {INTEREST_CLASSES["drinking"]}
EATING_RELATED_IDS = {INTEREST_CLASSES["eating"]}

# --- 6. 其他配置 ---
MINIMUM_FACE_SIZE = 60
CAMERA_ID = "CAM-01"

# --- 6. 采样与交互配置 (Enrollment & Interaction) ---
# --- 核心修改：添加缺失的配置项 ---
TARGET_SAMPLES = 20
SAMPLING_INSTRUCTIONS = [ 
    "1/6: 请正视前方", 
    "2/6: 请缓慢向左转头", 
    "3/6: 请缓慢向右转头", 
    "4/6: 请缓慢抬头", 
    "5/6: 请缓慢低头", 
    "6/6: 请张开嘴巴"
]
STAGE_DURATION = 3
MINIMUM_FACE_SIZE = 60 


DEBUG_CAPTURE_FRAMES = 100 # 在偵錯模式下，總共捕獲並分析的影格數量

