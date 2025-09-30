import cv2
import numpy as np
import onnxruntime as ort

# -------------------------------
# 1. 配置参数
# -------------------------------
MODEL_PATH ='pt/best.onnx'
IMAGE_PATH = "test_frame.jpg"  # 替换为你的测试图
CLASS_NAMES = ["person", "smoking", "drinking", "eating"]

# 输入尺寸（必须和训练时一致，YOLOv8/v11 通常是 640）
INPUT_SIZE = 640

# 置信度和 NMS 阈值（先放宽一点测试）
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# -------------------------------
# 2. 加载 ONNX 模型
# -------------------------------
print("🔍 正在加载模型...")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
    print("✅ 模型加载成功")
    print("🔧 使用的执行提供者:", session.get_providers())
except Exception as e:
    print("❌ 模型加载失败:", e)
    exit(1)

# 查看输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape

print(f"📥 输入: {input_name}, shape: {input_shape}")
print(f"📤 输出: {output_name}, shape: {output_shape}")

# -------------------------------
# 3. 图像预处理
# -------------------------------
def preprocess(image):
    h, w = image.shape[:2]
    # 等比缩放 + 黑边填充
    scale = INPUT_SIZE / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建 640x640 画布并居中粘贴
    padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    pad_w = (INPUT_SIZE - new_w) // 2
    pad_h = (INPUT_SIZE - new_h) // 2
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    # HWC -> CHW -> float32
    blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # (1, 3, 640, 640)
    return blob, scale, pad_w, pad_h

# -------------------------------
# 4. 后处理（解码输出）
# -------------------------------
# 简化版本的后处理逻辑
def postprocess(outputs, scale, pad_w, pad_h):
    # outputs: (1, 9, 8400) → (8400, 9)
    outputs = np.squeeze(outputs).transpose(1, 0)  # (8400, 9)

    boxes = []
    scores = []
    class_ids = []

    for row in outputs:
        x, y, w, h, obj_conf = row[:5]
        class_probs = row[5:9]  # 取 4 个类别的概率

        cls_id = np.argmax(class_probs)
        cls_conf = class_probs[cls_id]
        final_conf = obj_conf * cls_conf

        if final_conf < CONF_THRESHOLD:
            continue

        x1 = (x - w / 2 - pad_w) / scale
        y1 = (y - h / 2 - pad_h) / scale
        x2 = (x + w / 2 - pad_w) / scale
        y2 = (y + h / 2 - pad_h) / scale

        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        class_ids.append(cls_id)

    # NMS（修复 tuple 问题）
    if len(boxes) == 0:
        return []

    nms_result = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    indices = nms_result[0] if isinstance(nms_result, tuple) else nms_result

    result = []
    for i in indices.flatten():
        result.append({
            'box': [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])],
            'score': float(scores[i]),
            'class_id': class_ids[i],
            'class_name': CLASS_NAMES[class_ids[i]]
        })
    return result

# -------------------------------
# 5. 执行推理
# -------------------------------
print("📷 正在读取图像...")
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ 无法读取图像: {IMAGE_PATH}")
    exit(1)

blob, scale, pad_w, pad_h = preprocess(image)

print("🚀 正在推理...")
try:
    outputs = session.run([output_name], {input_name: blob})[0]
    print(f"✅ 推理成功，原始输出 shape: {outputs.shape}")
    
    # 打印前几行输出（调试用）
    print("🔍 原始输出前5行（[x,y,w,h,conf,cls0,cls1,...]）:")
    print(outputs[0, :5, :8])  # 显示前5个框的前8个值

    # 后处理
    results = postprocess(outputs, scale, pad_w, pad_h)

    if len(results) == 0:
        print("❌ 未检测到任何目标")
    else:
        print(f"✅ 检测到 {len(results)} 个目标:")
        for res in results:
            print(f"  📦 {res['class_name']} ({res['class_id']}): "
                  f"score={res['score']:.3f}, box={res['box']}")

        # 在图像上画框
        for res in results:
            x1, y1, x2, y2 = res['box']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{res['class_name']}: {res['score']:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite("output_debug.jpg", image)
        print("🖼️ 结果已保存为 output_debug.jpg")

except Exception as e:
    print("❌ 推理失败:", e)
    import traceback
    traceback.print_exc()