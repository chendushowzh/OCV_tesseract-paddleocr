import cv2
import numpy as np

def detect_text_blocks_east(image, east_model, min_confidence=0.5):
    """使用 EAST 文本检测器检测文本块"""
    # 读取图像尺寸
    orig = image.copy()
    (H, W) = image.shape[:2]

    # 定义输入图像的目标宽度和高度（EAST 模型要求的尺寸是 32 的倍数）
    newW, newH = (640, 640)
    rW = W / float(newW)
    rH = H / float(newH)

    # 调整图像大小以适应 EAST 模型的输入
    image = cv2.resize(image, (newW, newH))

    # 加载预训练的 EAST 模型
    net = cv2.dnn.readNet(east_model)

    # 构建输入 blob，并进行前向传播来获取预测
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # EAST 模型的输出层名字
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",  # 文字是否存在的概率图
        "feature_fusion/concat_3"  # 文字的几何框
    ]
    
    # 前向传播，获取两层输出
    (scores, geometry) = net.forward(layerNames)

    # 解析输出，获取文本块
    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)

    # 应用非极大值抑制来抑制重叠的边界框
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # 调整框的大小回原始图像尺度，并绘制框
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # 绘制文本块矩形框
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return orig

def decode_predictions(scores, geometry, min_confidence):
    """从 EAST 的输出中提取文本块的矩形框"""
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            # 解析几何框数据
            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """使用非极大值抑制来过滤重叠的边界框"""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# 使用 EAST 检测文本块
image = cv2.imread('D:/LU/assignment/cv/test1.jpg')
east_model = 'frozen_east_text_detection.pb'  # EAST 模型文件路径
output_image = detect_text_blocks_east(image, east_model)

cv2.imshow('Text Blocks - EAST', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

