import cv2
import pytesseract
import numpy as np
import re
from pytesseract import Output
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image

# 1. 图像预处理函数定义

# 灰度处理
def get_grayscale(image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# 降噪
def remove_noise(image):
    image= cv2.GaussianBlur(image, (3, 3), 0,0)
    return image

# 二值化
def thresholding(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return image 

# 膨胀
def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    image=cv2.dilate(image, kernel, iterations=1)
    return image

# 腐蚀
def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    image=cv2.erode(image, kernel, iterations=1)
    return image

# 开运算（先腐蚀后膨胀）
def opening(image):
    kernel = np.ones((1, 3), np.uint8)
    image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

#锐化
def sharpen(image):
    # 使用拉普拉斯算子检测图像边缘
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # 转换回 uint8 格式
    sharpened_image = cv2.convertScaleAbs(laplacian)
    return sharpened_image

# 边缘检测
def canny(image):
    image=cv2.Canny(image, 30, 140)
    return image

# 倾斜校正
def deskew(image): 
    # 边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # 存储检测到的角度
    angles = []

    # 如果检测到了直线
    if lines is not None and angles is not None:
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta)  # 将弧度转换为角度
            if angle > 45 and angle < 135:  # 过滤掉垂直和接近水平的线
                angles.append(angle - 90)  # 将角度调整为相对于水平线

    # 计算角度的中位数作为最终倾斜角
    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        median_angle = 0  # 如果没有找到合适的线，假设没有倾斜

    # 计算旋转矩阵并对图像进行旋转
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # 进行旋转并返回校正后的图像
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# 尺寸归一化 将图像的宽度归一化为指定大小，同时保持纵横比。
def resize_to_standard(image, width=800):
    ratio = width / image.shape[1]
    dim = (width, int(image.shape[0] * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# 透视矫正 根据四个点进行透视变换校正。
'''def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
'''
# 初始化框选的坐标
refPt = []
cropping = False

# 鼠标回调函数，用于获取框选的矩形区域
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    # 按下左键，记录起始坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 移动鼠标时，实时绘制矩形
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        image_copy = param.copy()
        cv2.rectangle(image_copy, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", image_copy)

    # 释放左键，记录结束坐标
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # 画出最终的矩形框
        cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", param)
        custom_config_multilang = r'-l chi+eng --psm 6'
        # 进行OCR
        roi = param[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        ocr_result = pytesseract.image_to_string(roi, config=custom_config_multilang)
        print("OCR Result for Selected Area:")
        print(ocr_result)


def draw_text_boxes(image):
    #识别并绘制文字框
    image_with_boxes=image.copy()
    if len(image.shape) == 3:
        h, w, _ = image.shape  # 彩色图像有3个维度
    else:
        h, w = image.shape  # 灰度图像有2个维度
    boxes = pytesseract.image_to_boxes(image_with_boxes,config=r'-l chi+eng --psm 6')
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(image_with_boxes, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    return image_with_boxes

def draw_confident_text_boxes(image, d):
    #绘制高置信度的文字框
    image_with_conf_boxes=image.copy()
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # 置信度大于60
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(image_with_conf_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_with_conf_boxes


def preprocess(image):
    #图像预处理     由于不同图片预处理后的ocr效果不同，只保留最简单的方法
    result = resize_to_standard(image)  # 尺寸归一化
    result = get_grayscale(result)        # 灰度处理
    #result = remove_noise(result)        # 去噪
    result = deskew(result)            # 倾斜校正  包含边缘检测
    #result =opening(result)           # 开运算
    #result = sharpen(result)          #锐化


    return result

# 选取pytesseract路径
pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract\tesseract.exe'
# 读取图像
image = cv2.imread('D:/LU/assignment/cv/test2.jpg')

cv2.imshow("origin",image)
# 图像预处理
image=preprocess(image)

# pytesseract部分
image_with_boxes = draw_text_boxes(image)
cv2.imshow('Text Boxes', image_with_boxes)
cv2.waitKey(0)

# 提取文本数据
d = pytesseract.image_to_data(image, output_type=Output.DICT,config=r'-l chi+eng --psm 6')
# 输出识别的文本
n_boxes = len(d['text'])  # 识别到的文本块数
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:  # 置信度大于60输出
        text = d['text'][i]
        conf = d['conf'][i]
        print(f"Text: {text}, Confidence: {conf}")

image_with_conf_boxes = draw_confident_text_boxes(image, d)
cv2.imshow('Confident Text Boxes', image_with_conf_boxes)
cv2.waitKey(0)

# Paddleocr部分
paddleocr = PaddleOCR(
    use_gpu=False,                # 使用CPU
    lang='ch',                    # 语言为中文
    use_angle_cls=True,           # 使用方向分类器
    det_db_thresh=0.1,            # 文本检测阈值
    det_db_box_thresh=0.3,        # 文本框阈值
    det_db_unclip_ratio=1.6,      # 调整文本框大小的比例
    rec_char_type='en',           # 识别字符类型为中文
    rec_image_shape="3, 32, 320", # 识别图像的输入形状
    cls_thresh=0.9                # 方向分类器阈值
)
result = paddleocr.ocr(image, cls=True)


# 可视化文本检测框和 OCR 结果
#image = Image.open(image).convert('RGB')
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 将灰度图像转换为 BGR 彩色图像,当对图片进行预处理后要用
boxes = [elements[0] for elements in result[0]]  # 提取检测到的文本框
txts = [elements[1][0] for elements in result[0]]  # 提取识别的文本内容
scores = [elements[1][1] for elements in result[0]]  # 提取识别的置信度

# 绘制检测到的文本框和识别结果
im_show = draw_ocr(image, boxes, txts, scores)
im_show = Image.fromarray(im_show)
plt.imshow(im_show)
plt.show() 




#手动截取部分

# 创建窗口并绑定鼠标事件
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop, param=image)

# 显示图像并等待用户框选
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

