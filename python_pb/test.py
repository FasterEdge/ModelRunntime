# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import tensorflow as tf
import time

# pb格式模型测试代码 By：tyza66

# 加载模型
def load_pb_model(pb_file_path):
    # 加载pb模型
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # # === 打印所有张量名（带 :0） ===
    # for op in graph.get_operations():
    #     for t in op.outputs:
    #         print(t.name)          # 看到真实输入/输出名再抄下来

    return graph

# 数据预处理
def preprocess_image(image_path, target_size):
    # 读取图像
    image = cv2.imread(image_path)
    original_shape = [image.shape[0], image.shape[1]]
    # 调整图像大小
    image = cv2.resize(image, target_size)
    # 转换为浮点型并归一化
    image = image.astype(np.float32) / 255.0
    # 扩展维度以匹配模型输入
    image = np.expand_dims(image, axis=0)
    return image, original_shape

# 运行模型进行推理
def run_inference(graph, input_data, original_shape):
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name('images:0')
        shapes_tensor = graph.get_tensor_by_name('shapes:0')
        output_classes = graph.get_tensor_by_name('concat_19:0')
        output_scores = graph.get_tensor_by_name('concat_18:0')
        output_boxes = graph.get_tensor_by_name('concat_17:0')
        
        start_time = time.time()
        classes, scores, boxes = sess.run(
            [output_classes, output_scores, output_boxes],
            feed_dict={
                input_tensor: input_data,
                shapes_tensor: original_shape
            }
        )
        inference_time = time.time() - start_time
        
    return (classes, scores, boxes), inference_time

# 绘制检测框
def draw_boxes(image_path, classes, scores, boxes, conf_threshold=0.5):
    image = cv2.imread(image_path)
    class_names = {0: 'person', 1: 'helmet', 2: 'helmet_on', 3: 'helmet_off'}
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (255, 0, 255)}
    
    for i in range(len(classes)):
        if scores[i] >= conf_threshold:
            class_id = int(classes[i])
            y1, x1, y2, x2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            color = colors.get(class_id, (0, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_names.get(class_id, 'other')}: {scores[i]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image



# 入口函数
if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("TensorFlow version:", tf.__version__)
    print("="*60)
    
    pb_file_path = "yolov3_resnet18.pb"
    image_path = "example2.png"
    
    print("加载模型中...")
    load_start = time.time()
    graph = load_pb_model(pb_file_path)
    load_time = time.time() - load_start
    print(f"模型加载耗时: {load_time:.4f} 秒")
    print("="*60)
    
    print("预处理图像中...")
    preprocess_start = time.time()
    input_data, original_shape = preprocess_image(image_path, target_size=(736, 416))
    preprocess_time = time.time() - preprocess_start
    print(f"预处理耗时: {preprocess_time:.4f} 秒")
    print(f"原始图像尺寸: {original_shape}")
    print("="*60)
    
    warmup_runs = 3
    test_runs = 10
    
    print(f"预热推理 {warmup_runs} 次...")
    for i in range(warmup_runs):
        _, warmup_time = run_inference(graph, input_data, original_shape)
        print(f"  预热 {i+1}: {warmup_time:.4f} 秒")
    print("="*60)
    
    print(f"正式测试 {test_runs} 次推理...")
    inference_times = []
    for i in range(test_runs):
        (classes, scores, boxes), inference_time = run_inference(graph, input_data, original_shape)
        inference_times.append(inference_time)
        print(f"  测试 {i+1}: {inference_time:.4f} 秒")
        print(f"    检测到目标数量: {len(classes)}")
        if len(classes) > 0:
            print(f"    类别: {classes[:5]}")
            print(f"    置信度: {scores[:5]}")
            print(f"    框坐标shape: {boxes.shape}")
    
    print("="*60)
    print("性能统计:")
    print(f"  平均耗时: {np.mean(inference_times):.4f} 秒")
    print(f"  最小耗时: {np.min(inference_times):.4f} 秒")
    print(f"  最大耗时: {np.max(inference_times):.4f} 秒")
    print(f"  标准差:   {np.std(inference_times):.4f} 秒")
    print(f"  FPS:      {1.0/np.mean(inference_times):.2f}")
    print("="*60)
    
    print("绘制检测框并保存结果...")
    result_image = draw_boxes(image_path, classes, scores, boxes)
    output_path = "result_" + image_path
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")
    print("="*60)


