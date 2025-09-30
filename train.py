# train.py
from ultralytics import YOLO
import os

# 设置环境变量（可选：防止一些显存问题）
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    # 加载预训练模型（官方 YOLOv8n）
    model = YOLO('yolo11n.pt')  # 或 yolov8s.pt 如果你想要更高精度

    # 开始训练
    results = model.train(
        data='super_dataset/data.yaml',      # 数据集配置文件
        epochs=100,               # 训练轮数
        imgsz=640,                # 输入图像大小
        batch=32,                 # 批次大小（根据显存调整：8/16/32）
        name='yolo11n-all',       # 实验名称
        project='runs',           # 保存路径
        exist_ok=False,           # 禁止覆盖已有实验
        patience=20,              # 早停：20轮无提升则停止
        save_period=1,            # 每轮都保存
        device=0,                 # GPU ID，0 表示第一块 GPU
        workers=16,                # 数据加载线程数
        seed=42,                  # 随机种子，保证可复现
        optimizer='AdamW',        # 优化器
        lr0=0.001,                # 初始学习率
        lrf=0.1,                  # 最终学习率 = lr0 * lrf
        momentum=0.937,           # SGD/Adam 动量
        weight_decay=0.0005,      # 权重衰减
        warmup_epochs=3.0,        # 学习率预热
        box=7.5,                  # 检测框损失权重
        cls=0.5,                  # 分类损失权重
        dfl=1.5,                  # 分布焦点损失
        hsv_h=0.015,              # 数据增强：色相
        hsv_s=0.7,                # 饱和度
        hsv_v=0.4,                # 明度
        degrees=0.0,              # 旋转
        translate=0.1,            # 平移
        scale=0.5,                # 缩放
        shear=0.0,                # 剪切
        flipud=0.0,               # 上下翻转
        fliplr=0.5,               # 左右翻转
        bgr=0.0,                  # BGR 概率
        mosaic=1.0,               # 马赛克增强
        mixup=0.0,                # MixUp
        copy_paste=0.0,           # Copy-Paste
    )

    print("✅ 训练完成！")