# #file:/workspace/export_model.py
# import torch
# from ultralytics import YOLO

# def export_model():
#     print("加载模型...")
#     # 加载训练好的模型
#     model = YOLO('pt/yolov11n-all.pt')  # 替换为您的实际模型路径
    
    # export.py
# from ultralytics import YOLO

# # 加载你的模型
# model = YOLO("pt/yolov11n-all.pt")  # 自动识别结构

from ultralytics import YOLO

# 使用修复后的模型
model = YOLO("models/hat.pt")

# 导出（必须 simplify=True）
model.export(
    format="onnx",
    imgsz=640,
    opset=13,
    simplify=True,      # ← 关键！
    dynamic=False,
    device="cpu"
)