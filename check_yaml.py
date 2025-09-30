import cv2
import numpy as np
import onnxruntime as ort

# 假设你已经做了推理
# output = session.run(None, {'images': input_data})  # shape: (1, 8, 8400)

def decode_yolov8_output(output, conf_threshold=0.25, iou_threshold=0.45):
    """
    解码 YOLOv8 的原始输出
    """
    # output shape: (1, 8, 8400)
    output = output[0]  # 去掉 batch 维度 -> (8, 8400)
    output = output.T    # 转置 -> (8400, 8)
    
    # 分离坐标、置信度、类别
    xywh = output[:, :4]      # 中心坐标 + 宽高
    conf = output[:, 4]       # 目标置信度
    cls_conf = output[:, 5:]  # 类别置信度

    # 计算总置信度 = obj_conf * max_cls_conf
    scores = conf * np.max(cls_conf, axis=1)  # (8400,)
    
    # 置信度过滤
    mask = scores > conf_threshold
    if not np.any(mask):
        return []
    
    xywh = xywh[mask]
    scores = scores[mask]
    labels = np.argmax(cls_conf[mask], axis=1)

    # xywh -> xyxy
    boxes = np.zeros_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2

    # NMS
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    
    if len(keep) == 0:
        return []
    
    keep = keep.flatten() if hasattr(keep, 'flatten') else keep
    result = []
    for i in keep:
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        result.append({
            'box': box,      # [x1, y1, x2, y2]
            'score': score,
            'label': label
        })
    
    return result

# ✅ 使用示例
raw_output = output  # 来自 ONNX 推理
detections = decode_yolov8_output(raw_output, conf_threshold=0.25)

if len(detections) > 0:
    print(f"✅ 检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"  类别: {det['label']}, 置信度: {det['score']:.3f}, 位置: {det['box']}")
else:
    print("❌ 未检测到目标")