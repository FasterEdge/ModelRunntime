import cv2
import numpy as np
import onnxruntime as ort
import time

# onnx格式模型测试代码 By：tyza66

# 加载模型
def load_onnx_model(onnx_file_path):
    session = ort.InferenceSession(onnx_file_path)
    return session

# 数据预处理
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    original_shape = [image.shape[0], image.shape[1]]
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, original_shape

# NMS非极大值抑制
def nms(boxes, scores, class_ids, iou_threshold=0.45):
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        return (
            [class_ids[i] for i in indices],
            [scores[i] for i in indices],
            [boxes[i] for i in indices]
        )
    return [], [], []

# 后处理YOLOv5输出
def postprocess_yolo(outputs, original_shape, input_shape, conf_threshold=0.25, iou_threshold=0.45, debug=False):
    predictions = outputs[0]
    
    if debug:
        print(f"输出shape: {predictions.shape}")
    
    boxes = []
    scores = []
    class_ids = []
    
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    
    orig_h, orig_w = original_shape
    input_h, input_w = input_shape
    
    for detection in predictions:
        confidence = detection[4]
        if confidence > conf_threshold:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            final_confidence = confidence * class_confidence
            
            if final_confidence > conf_threshold:
                x_center, y_center, width, height = detection[0:4]
                
                x1 = (x_center - width / 2) * orig_w / input_w
                y1 = (y_center - height / 2) * orig_h / input_h
                x2 = (x_center + width / 2) * orig_w / input_w
                y2 = (y_center + height / 2) * orig_h / input_h
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(final_confidence))
                class_ids.append(int(class_id))
    
    if debug:
        print(f"NMS前检测到 {len(class_ids)} 个目标")
    
    class_ids, scores, boxes = nms(boxes, scores, class_ids, iou_threshold)
    
    if debug:
        print(f"NMS后检测到 {len(class_ids)} 个目标")
        if len(boxes) > 0:
            print(f"第一个框坐标: {boxes[0]}")
    
    return class_ids, scores, boxes

# 绘制检测框
def draw_boxes(image_path, classes, scores, boxes, conf_threshold=0.5):
    image = cv2.imread(image_path)
    print(f"图像尺寸: {image.shape}")
    
    coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    np.random.seed(42)
    colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(len(coco_names))}
    
    drawn_count = 0
    for i in range(len(classes)):
        if scores[i] >= conf_threshold:
            class_id = int(classes[i])
            x1, y1, x2, y2 = boxes[i]
            print(f"框 {i}: 原始坐标 ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), 类别: {class_id}, 置信度: {scores[i]:.3f}")
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            color = colors.get(class_id, (0, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{coco_names[class_id] if class_id < len(coco_names) else 'unknown'}: {scores[i]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            drawn_count += 1
    
    print(f"实际绘制了 {drawn_count} 个框")
    return image

# 运行模型进行推理
def run_inference(session, input_data, original_shape, input_shape, debug=False):
    input_name = session.get_inputs()[0].name
    
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = time.time() - start_time
    
    class_ids, scores, boxes = postprocess_yolo(outputs, original_shape, input_shape, debug=debug)
    
    return (class_ids, scores, boxes), inference_time

# 入口函数
if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("ONNXRuntime version:", ort.__version__)
    print("="*60)
    
    onnx_file_path = "yolov5n.onnx"
    image_path = "example2.png"
    
    print("加载模型中...")
    load_start = time.time()
    session = load_onnx_model(onnx_file_path)
    load_time = time.time() - load_start
    print(f"模型加载耗时: {load_time:.4f} 秒")
    print("="*60)
    
    print("预处理图像中...")
    preprocess_start = time.time()
    input_data, original_shape = preprocess_image(image_path, target_size=(640, 640))
    preprocess_time = time.time() - preprocess_start
    print(f"预处理耗时: {preprocess_time:.4f} 秒")
    print(f"原始图像尺寸: {original_shape}")
    print("="*60)
    
    warmup_runs = 3
    test_runs = 10
    input_shape = (640, 640)
    
    print(f"预热推理 {warmup_runs} 次...")
    for i in range(warmup_runs):
        _, warmup_time = run_inference(session, input_data, original_shape, input_shape, debug=(i==0))
        print(f"  预热 {i+1}: {warmup_time:.4f} 秒")
    print("="*60)
    
    print(f"正式测试 {test_runs} 次推理...")
    inference_times = []
    for i in range(test_runs):
        (class_ids, scores, boxes), inference_time = run_inference(session, input_data, original_shape, input_shape)
        inference_times.append(inference_time)
        print(f"  测试 {i+1}: {inference_time:.4f} 秒")
        print(f"    检测到目标数量: {len(class_ids)}")
        if len(class_ids) > 0:
            print(f"    类别: {class_ids[:5]}")
            print(f"    置信度: {[f'{s:.3f}' for s in scores[:5]]}")
            print(f"    框坐标数量: {len(boxes)}")
    
    print("="*60)
    print("性能统计:")
    print(f"  平均耗时: {np.mean(inference_times):.4f} 秒")
    print(f"  最小耗时: {np.min(inference_times):.4f} 秒")
    print(f"  最大耗时: {np.max(inference_times):.4f} 秒")
    print(f"  标准差:   {np.std(inference_times):.4f} 秒")
    print(f"  FPS:      {1.0/np.mean(inference_times):.2f}")
    print("="*60)
    
    print("绘制检测框并保存结果...")
    result_image = draw_boxes(image_path, class_ids, scores, boxes)
    output_path = "result_" + image_path
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")
    print("="*60)
