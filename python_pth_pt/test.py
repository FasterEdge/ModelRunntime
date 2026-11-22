# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
# PyTorch 权重（.pt / .pth）目标检测推理示例
#
# yolov5s.pt 是 ultralytics YOLOv5 的 checkpoint（pickle 里保存了 DetectionModel
# 实例），因此加载时必须能 import 到 yolov5 仓库的 `models` 包。两种方式任选其一：
#   1) git clone https://github.com/ultralytics/yolov5 && export YOLOV5_DIR=<仓库路径>
#   2) pip install yolov5（本脚本会自动把 site-packages/yolov5 加入 sys.path）
# 缺这个包时脚本会明确报错退出，不会静默返回空结果。

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import torch

# ===== 动态参数（按实际模型修改）=====
# WEIGHT_PATH: checkpoint 路径（YOLOv5 的 .pt；纯 state_dict 的 .pth 需自行定义模型结构）
WEIGHT_PATH = "yolov5s.pt"          # TODO: 替换为你的 .pt / .pth 路径
# IMAGE_PATH: 输入图片
IMAGE_PATH = "example.png"          # TODO: 替换输入图片
# INPUT_SIZE: 网络输入尺寸 (W, H)
INPUT_SIZE = (640, 640)             # TODO: 按模型输入尺寸修改
# DEVICE: "cpu" / "cuda:0"
DEVICE = "cpu"                      # TODO: 按可用设备修改
# CONF_THRESHOLD / IOU_THRESHOLD: 后处理阈值
CONF_THRESHOLD = 0.25               # TODO: 按需修改
IOU_THRESHOLD = 0.45                # TODO: 按需修改
# NMS_CLASSES: 800 = 各类别独立 NMS（YOLOv5 默认），0 = class-agnostic
NMS_CLASSES = 800


def _ensure_models_importable() -> None:
    """让 yolov5 checkpoint 需要的 `models` 包可 import，否则给出可执行的报错。"""
    try:
        import models  # noqa: F401
        return
    except ImportError:
        pass

    repo = os.environ.get("YOLOV5_DIR")
    if repo and os.path.isdir(repo):
        sys.path.insert(0, repo)
        import models  # noqa: F401
        return

    try:
        import yolov5
    except ImportError:
        raise RuntimeError(
            "加载 YOLOv5 checkpoint 需要 yolov5 仓库的 `models` 包。请二选一：\n"
            "  git clone https://github.com/ultralytics/yolov5 并设置 YOLOV5_DIR\n"
            "  pip install yolov5"
        ) from None

    sys.path.insert(0, os.path.dirname(yolov5.__file__))
    import models  # noqa: F401


def load_model(weight_path: str, device: str):
    # 安全注意: torch.load 仍是 pickle 反序列化, 恶意 .pt 可在加载时执行任意代码(RCE),
    # 因此 weights_only=False 只对完全可信的权重启用（yolov5 checkpoint 含模型类实例,
    # 无法用 weights_only=True 加载）。未知来源权重请先 weights_only=True 只取 state_dict。
    _ensure_models_importable()
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model.float().to(device).eval()
    return model


def preprocess(image_path: str, input_size: tuple[int, int], device: str):
    """BGR->RGB, resize, /255, HWC->CHW, 加 batch 维（YOLOv5 标准预处理）。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    original_shape = (img.shape[0], img.shape[1])
    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return torch.from_numpy(np.ascontiguousarray(img)).to(device), original_shape


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """单类别 NMS（IoU 贪心抑制），只用 numpy 以便结果可复现。"""
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size > 0:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return kept


def postprocess(outputs, original_shape, input_size,
                conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                max_det=300, nms_classes=NMS_CLASSES):
    """YOLOv5 推理输出 (1, N, 5+num_classes) -> (boxes, scores, class_ids)。

    前 4 列是 cx,cy,w,h（输入尺寸像素坐标），第 5 列是 objectness，
    其后是各类别分数；最终置信度 = objectness * class_score。
    """
    pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    # 训练模式会返回 (pred, raw_feature_maps)，这里只取 decode 后的 pred
    pred = pred[0].detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred[0])
    if pred.ndim == 3:
        pred = pred[0]

    keep = pred[:, 4] > conf_threshold
    pred = pred[keep]
    if pred.shape[0] == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)

    class_scores = pred[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = pred[:, 4] * class_scores.max(axis=1)
    valid = scores > conf_threshold
    pred, scores, class_ids = pred[valid], scores[valid], class_ids[valid]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
    # 还原到原图尺寸
    orig_h, orig_w = original_shape
    in_w, in_h = input_size
    boxes[:, [0, 2]] *= orig_w / in_w
    boxes[:, [1, 3]] *= orig_h / in_h

    final_boxes, final_scores, final_ids = [], [], []
    unique_classes = np.unique(class_ids) if nms_classes > 0 else [None]
    for cls in unique_classes:
        if cls is None:
            idx = np.arange(boxes.shape[0])
        else:
            idx = np.where(class_ids == cls)[0]
        if idx.size == 0:
            continue
        kept = nms(boxes[idx], scores[idx], iou_threshold)[:max_det]
        final_boxes.append(boxes[idx][kept])
        final_scores.append(scores[idx][kept])
        final_ids.append(class_ids[idx][kept])
    if not final_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    return (np.concatenate(final_boxes), np.concatenate(final_scores),
            np.concatenate(final_ids))


def run_inference(model, data):
    with torch.no_grad():
        return model(data)


if __name__ == "__main__":
    model = load_model(WEIGHT_PATH, DEVICE)
    data, original_shape = preprocess(IMAGE_PATH, INPUT_SIZE, DEVICE)
    print(f"device: {DEVICE}  输入: {IMAGE_PATH} {original_shape} -> {tuple(data.shape)}")

    for _ in range(3):
        run_inference(model, data)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.time()
    outputs = run_inference(model, data)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"inference time: {elapsed:.4f}s  ({1.0 / elapsed:.2f} FPS)")

    raw = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    print(f"output shape: {tuple(raw.shape)}")

    boxes, scores, class_ids = postprocess(outputs, original_shape, INPUT_SIZE)
    names = getattr(model, "names", {})
    print(f"检测到 {len(boxes)} 个目标")
    for box, score, cls in zip(boxes, scores, class_ids):
        label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
        print(f"  {label}: {score:.3f}  box=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
