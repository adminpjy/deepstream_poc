# quad_model_optimized.py - 使用四个模型分别检测人形、吸烟行为、安全帽和打电话（优化流程）
import cv2
import numpy as np
import config
from ultralytics import YOLO
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_box_center(box):
    """获取边界框的中心点"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_box_area(box):
    """计算边界框的面积"""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def is_box_inside(inner_box, outer_box, threshold=0.5):
    """检查inner_box是否在outer_box内部，通过重叠面积比例判断"""
    x1_min, y1_min, x1_max, y1_max = inner_box
    x2_min, y2_min, x2_max, y2_max = outer_box
    
    # 计算重叠区域
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)
    
    # 如果没有重叠，返回False
    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        return False
    
    # 计算重叠面积
    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
    inner_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    # 如果重叠面积占内部框面积的比例超过阈值，则认为在内部
    return (overlap_area / inner_area) >= threshold if inner_area > 0 else False

def expand_box(box, expansion_ratio=0.0):
    """扩展边界框，用于扩大检测范围"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # 扩展边界
    expand_x = int(width * expansion_ratio)
    expand_y = int(height * expansion_ratio)
    
    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        x2 + expand_x,
        y2 + expand_y
    )

def calculate_distance(box1_center, box2_center):
    """计算两个边界框中心点之间的距离"""
    return np.sqrt((box1_center[0] - box2_center[0])**2 + (box1_center[1] - box2_center[1])**2)

def validate_model():
    """
    使用OpenCV显示窗口验证四模型优化检测流程
    """
    # 加载人员检测模型
    logger.info("正在加载人员检测模型...")
    try:
        person_detector = YOLO(config.PERSON_MODEL)  # 人员检测模型
        logger.info("✅ 人员检测模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 人员检测模型加载失败: {e}")
        return

    # 加载吸烟检测模型
    logger.info("正在加载吸烟检测模型...")
    try:
        smoking_detector = YOLO(config.SMOKING_MODEL)  # 吸烟检测模型
        logger.info("✅ 吸烟检测模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 吸烟检测模型加载失败: {e}")
        return

    # 加载安全帽检测模型
    logger.info("正在加载安全帽检测模型...")
    try:
        hat_detector = YOLO(config.HAT_MODEL)  # 安全帽检测模型
        logger.info("✅ 安全帽检测模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 安全帽检测模型加载失败: {e}")
        return

    # 加载打电话检测模型
    logger.info("正在加载打电话检测模型...")
    try:
        phone_detector = YOLO(config.PHONE_MODEL)  # 打电话检测模型
        logger.info("✅ 打电话检测模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 打电话检测模型加载失败: {e}")
        return

    # 初始化摄像头
    logger.info("正在启动摄像头...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        logger.error("❌ 无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    logger.info("✅ 摄像头已启动")
    logger.info("按 'q' 退出, 按 's' 保存当前帧")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("无法从摄像头读取帧")
            continue
            
        frame_count += 1
        
        # 创建显示帧的副本
        display_frame = frame.copy()
        
        try:
            # 第一步：使用人员检测模型检测人员（只检测person类别）
            logger.debug("正在进行人员检测...")
            person_results = person_detector.predict(
                frame, 
                verbose=False, 
                conf=config.PERSON_CONF_THRESHOLD,
                classes=[config.INTEREST_CLASSES["person"]]  # 只检测person类别
            )
            person_result = person_results[0]
            
            # 存储检测到的人员框
            persons = []
            
            # 处理人员检测结果
            if person_result.boxes is not None:
                for i, box in enumerate(person_result.boxes):
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    confidence = float(box.conf[0])
                    
                    # 绘制人员边界框（蓝色）
                    color = (255, 0, 0)  # 蓝色
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label_text = f"Person {i+1}: {confidence:.2f}"
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), 
                                 (x1 + text_size[0], y1), color, -1)
                    cv2.putText(display_frame, label_text, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 保存人员框信息，扩展人员框以包含周围区域
                    expanded_box = expand_box((x1, y1, x2, y2), 0.0)  # 不扩展，保持原框
                    persons.append({
                        'id': i+1,
                        'box': (x1, y1, x2, y2),
                        'expanded_box': expanded_box,
                        'confidence': confidence,
                        'has_hat': False,
                        'is_smoking': False,
                        'is_using_phone': False
                    })
            
            # 只有检测到人员时才进行其他检测
            if persons:
                # 第二步：使用吸烟检测模型检测吸烟行为（只检测smoking类别）
                logger.debug("检测到人员，正在进行吸烟检测...")
                smoking_results = smoking_detector.predict(
                    frame, 
                    verbose=False, 
                    conf=config.PERSON_CONF_THRESHOLD,
                    classes=[0]  # 只检测smoking类别
                )
                smoking_result = smoking_results[0]
                
                # 第三步：使用安全帽检测模型检测安全帽
                logger.debug("检测到人员，正在进行安全帽检测...")
                hat_results = hat_detector.predict(
                    frame, 
                    verbose=False, 
                    conf=config.PERSON_CONF_THRESHOLD
                )
                hat_result = hat_results[0]
                
                # 第四步：使用打电话检测模型检测打电话行为
                logger.debug("检测到人员，正在进行打电话检测...")
                phone_results = phone_detector.predict(
                    frame, 
                    verbose=False, 
                    conf=config.PERSON_CONF_THRESHOLD
                )
                phone_result = phone_results[0]
                
                # 处理吸烟检测结果 - 将吸烟行为与具体人员关联
                if smoking_result.boxes is not None:
                    for box in smoking_result.boxes:
                        sx1, sy1, sx2, sy2 = [int(coord) for coord in box.xyxy[0]]
                        confidence = float(box.conf[0])
                        smoking_box = (sx1, sy1, sx2, sy2)
                        smoking_center = get_box_center(smoking_box)
                        
                        # 查找与吸烟行为最相关的人员
                        best_person = None
                        min_distance = float('inf')
                        
                        for person in persons:
                            person_box = person['box']
                            person_center = get_box_center(person_box)
                            
                            # 检查吸烟框是否在人员框内或附近
                            if is_box_inside(smoking_box, person_box) or \
                               is_box_inside(smoking_box, person['expanded_box']):
                                # 计算距离
                                distance = calculate_distance(smoking_center, person_center)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_person = person
                        
                        # 如果找到相关联的人员，则标记该人员在吸烟
                        if best_person:
                            best_person['is_smoking'] = True
                            
                            # 在人员框上添加吸烟标签
                            px1, py1, px2, py2 = best_person['box']
                            smoking_label = f"Person {best_person['id']} smoking!"
                            cv2.putText(display_frame, smoking_label, (px1, py2 + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # 绘制吸烟行为边界框（红色）
                        color = (0, 0, 255)  # 红色
                        cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), color, 2)
                        
                        # 绘制标签
                        label_text = f"Smoking: {confidence:.2f}"
                        if not best_person:
                            label_text += " (Not associated with person)"
                        
                        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(display_frame, (sx1, sy1 - text_size[1] - 10), 
                                     (sx1 + text_size[0], sy1), color, -1)
                        cv2.putText(display_frame, label_text, (sx1, sy1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 处理安全帽检测结果 - 将安全帽与具体人员关联
                if hat_result.boxes is not None:
                    for box in hat_result.boxes:
                        hx1, hy1, hx2, hy2 = [int(coord) for coord in box.xyxy[0]]
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        hat_box = (hx1, hy1, hx2, hy2)
                        hat_center = get_box_center(hat_box)
                        
                        # 获取标签名
                        if class_id == 0:  # 假设hat类别ID为0
                            label = "Hat"
                            color = (255, 255, 0)  # 青色
                        elif class_id == 1:  # 假设person类别ID为1（安全帽模型中的人员）
                            label = "nohat"
                            color = (255, 0, 255)  # 紫色
                        else:
                            continue  # 跳过其他类别
                        
                        # 查找与安全帽最相关的人员
                        best_person = None
                        min_distance = float('inf')
                        
                        for person in persons:
                            person_box = person['box']
                            person_center = get_box_center(person_box)
                            
                            # 检查安全帽框是否在人员框内或附近
                            if is_box_inside(hat_box, person_box) or \
                               is_box_inside(hat_box, person['expanded_box']):
                                # 计算距离
                                distance = calculate_distance(hat_center, person_center)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_person = person
                        
                        # 如果找到相关联的人员，则更新该人员的安全帽状态
                        if best_person:
                            if label == "Hat":
                                best_person['has_hat'] = True
                                # 在人员框上添加安全帽标签
                                px1, py1, px2, py2 = best_person['box']
                                hat_label = f"Person {best_person['id']} has hat!"
                                cv2.putText(display_frame, hat_label, (px1, py2 + 50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            else:
                                best_person['has_hat'] = False
                        
                        # 绘制安全帽/人员边界框
                        cv2.rectangle(display_frame, (hx1, hy1), (hx2, hy2), color, 2)
                        
                        # 绘制标签
                        label_text = f"{label}: {confidence:.2f}"
                        if not best_person:
                            label_text += " (Not associated with person)"
                        
                        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(display_frame, (hx1, hy1 - text_size[1] - 10), 
                                     (hx1 + text_size[0], hy1), color, -1)
                        cv2.putText(display_frame, label_text, (hx1, hy1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 处理打电话检测结果 - 将打电话行为与具体人员关联
                if phone_result.boxes is not None:
                    for box in phone_result.boxes:
                        px1, py1, px2, py2 = [int(coord) for coord in box.xyxy[0]]
                        confidence = float(box.conf[0])
                        phone_box = (px1, py1, px2, py2)
                        phone_center = get_box_center(phone_box)
                        
                        # 查找与打电话行为最相关的人员
                        best_person = None
                        min_distance = float('inf')
                        
                        for person in persons:
                            person_box = person['box']
                            person_center = get_box_center(person_box)
                            
                            # 检查打电话框是否在人员框内或附近
                            if is_box_inside(phone_box, person_box) or \
                               is_box_inside(phone_box, person['expanded_box']):
                                # 计算距离
                                distance = calculate_distance(phone_center, person_center)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_person = person
                        
                        # 如果找到相关联的人员，则标记该人员在打电话
                        if best_person:
                            best_person['is_using_phone'] = True
                            
                            # 在人员框上添加打电话标签
                            person_px1, person_py1, person_px2, person_py2 = best_person['box']
                            phone_label = f"Person {best_person['id']} using phone!"
                            cv2.putText(display_frame, phone_label, (person_px1, person_py2 + 75),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # 绘制打电话行为边界框（黄色）
                        color = (0, 255, 255)  # 黄色
                        cv2.rectangle(display_frame, (px1, py1), (px2, py2), color, 2)
                        
                        # 绘制标签
                        label_text = f"Phone: {confidence:.2f}"
                        if not best_person:
                            label_text += " (Not associated with person)"
                        
                        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(display_frame, (px1, py1 - text_size[1] - 10), 
                                     (px1 + text_size[0], py1), color, -1)
                        cv2.putText(display_frame, label_text, (px1, py1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示帧计数和说明
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Quad Model Detection: Person + Smoking + Hat + Phone", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Blue: Person, Red: Smoking, Cyan: Hat, Purple: NoHat, Yellow: Phone", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save frame", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 显示结果
        cv2.imshow('Quad Model Person+Smoking+Hat+Phone Detection', display_frame)
        
        # 处理按键输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("用户退出")
            break
        elif key == ord('s'):
            # 保存当前帧
            filename = f"quad_model_detection_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"帧已保存为 {filename}")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    logger.info("程序已退出")

if __name__ == "__main__":
    validate_model()