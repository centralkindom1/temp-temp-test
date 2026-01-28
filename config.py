# config.py
import os
import sys
import pytesseract

# --- Tesseract 路径配置 (适配 Win7 和您的路径习惯) ---
TESSERACT_CMD = r'D:\Python\Scripts\tesseract.exe'
TESSDATA_DIR = r'D:\Python\Scripts\tessdata'

def setup_ocr_env():
    """初始化 OCR 环境"""
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    else:
        # 如果找不到指定路径，尝试系统 PATH
        pass

    if os.path.exists(TESSDATA_DIR):
        os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

# 初始化
setup_ocr_env()

# --- 业务逻辑配置 ---
class RAGConfig:
    # 标题判定阈值：默认比正文大 2 像素/磅
    HEADER_SIZE_THRESHOLD = 2.0 
    # 默认 OCR 开关
    DEFAULT_ENABLE_OCR = True 
    # 扫描分辨率
    OCR_DPI = 300