# train.py
from ultralytics import YOLO
import os

# 设置环境变量（可选：防止一些显存问题）
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    # 加载预训练模型（官方 YOLOv8n）
    model = YOLO('yolo11n.pt')  # 或 yolov8s.pt 如果你想要更高精度

    results = model.train(
        data='sources/smoking_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,
        name='yolo11n-all',
        project='runs',
        exist_ok=False,
        patience=30,
        save_period=1,
        device=0,
        workers=16,
        seed=42,
        optimizer='AdamW',
        lr0=0.003,           # ↑
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.00025,  # ↓
        warmup_epochs=3,
        box=5.5,             # ↓
        cls=0.35,            # ↓+
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,
        translate=0.1,
        scale=0.5,
        shear=0,
        flipud=0,
        fliplr=0.5,
        bgr=0,
        mosaic=0.9,
        mixup=0.15,          # ↑
        copy_paste=0.25,     # ↑
        close_mosaic=10      # 新增：最后10 epoch关闭mosaic
    )