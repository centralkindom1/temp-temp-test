# image_preprocessing.py
import cv2
import numpy as np
from PIL import Image

def preprocess_image_for_ocr(pil_image: Image):
    """
    图像预处理流水线：
    1. 转灰度
    2. 降噪
    3. 二值化
    """
    # 将 PIL Image 转换为 OpenCV 格式 (RGB -> BGR)
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 转换为灰度图
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # 中值滤波降噪 (去除椒盐噪声)
    gray = cv2.medianBlur(gray, 3)
    
    # OTSU 二值化 (自动寻找最佳阈值)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image