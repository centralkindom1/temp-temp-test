import pdfplumber
import pytesseract
import pandas as pd
import re
from PIL import Image
from config import RAGConfig
from image_preprocessing import preprocess_image_for_ocr

class DocumentLine:
    """定义一行文本及其属性"""
    def __init__(self, text, font_size, is_bold=False, page_num=0):
        self.text = text.strip()
        self.font_size = float(font_size)
        self.is_bold = is_bold
        self.page_num = page_num
        self.role = "BODY" # BODY, H1, H2

    def __repr__(self):
        return f"[{self.role}] size={self.font_size:.1f} | {self.text[:20]}..."

class PDFStructureParser:
    def __init__(self, filepath, use_ocr=True):
        self.filepath = filepath
        self.use_ocr = use_ocr
        self.parsed_lines = []
        self.body_font_size = 10.5 

    def parse(self, callback_signal=None):
        """执行解析主流程"""
        raw_lines = []
        
        with pdfplumber.open(self.filepath) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                if callback_signal:
                    callback_signal.emit(f"正在分析第 {page_num}/{total_pages} 页...", int(page_num/total_pages*50))
                
                if self.use_ocr:
                    # 优化1: 提高 OCR 清晰度，解决错别字
                    lines = self._extract_via_ocr(page, page_num, resolution=400)
                else:
                    lines = self._extract_via_plumber(page, page_num)
                
                raw_lines.extend(lines)

        self.parsed_lines = raw_lines
        
        # 步骤 1: 统计正文字号
        self._analyze_font_statistics()
        
        # 步骤 2: 初步打标 (H1/H2/BODY)
        self._tag_roles()
        
        # 步骤 3: 深度清洗与合并 (新增核心逻辑)
        self.parsed_lines = self._clean_and_merge(self.parsed_lines)
        
        return self.parsed_lines

    def _extract_via_ocr(self, page, page_num, resolution=400):
        """OCR 模式提取"""
        # 提高 DPI 有助于识别 '国际' vs '国破'
        img_pil = page.to_image(resolution=resolution).original
        processed_img = preprocess_image_for_ocr(img_pil)
        
        data = pytesseract.image_to_data(processed_img, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
        df = pd.DataFrame(data)
        
        # 过滤空文本
        df = df[df['text'].str.strip() != '']
        if df.empty: return []

        # 简单的行聚合：利用 block_num 和 line_num
        df['line_unique_id'] = df['block_num'].astype(str) + '_' + df['line_num'].astype(str)
        
        extracted_lines = []
        grouped = df.groupby('line_unique_id')
        
        for _, group in grouped:
            # 文本拼接
            text_part = "".join(group['text'].tolist())
            
            # 优化2: 基础清洗，去除 OCR 常见的行首噪点 (如 "7:", "B...")
            # 正则含义：去除行首的非中文字符杂质，如果它们后面跟着中文
            text_part = re.sub(r'^[A-Za-z0-9:._\-\s]{1,4}(?=[\u4e00-\u9fa5])', '', text_part)
            
            # 计算高度 (字号)
            avg_height = group['height'].mean()
            # 根据 DPI 缩放高度，使其数值更像常规字号 (便于理解)
            normalized_size = avg_height * (72 / resolution) * 2.5 # 经验系数
            
            extracted_lines.append(DocumentLine(text_part, normalized_size, page_num=page_num))
            
        return extracted_lines

    def _extract_via_plumber(self, page, page_num):
        """原生解析模式 (不变)"""
        lines = []
        words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
        if not words: return []
        
        current_top = words[0]['top']
        current_line_words = []
        
        for w in words:
            if abs(w['top'] - current_top) > 5:
                if current_line_words:
                    text = "".join([cw['text'] for cw in current_line_words])
                    avg_size = sum([float(cw['bottom']-cw['top']) for cw in current_line_words]) / len(current_line_words)
                    lines.append(DocumentLine(text, avg_size, page_num=page_num))
                current_line_words = [w]
                current_top = w['top']
            else:
                current_line_words.append(w)
                
        if current_line_words:
            text = "".join([cw['text'] for cw in current_line_words])
            avg_size = sum([float(cw['bottom']-cw['top']) for cw in current_line_words]) / len(current_line_words)
            lines.append(DocumentLine(text, avg_size, page_num=page_num))
            
        return lines

    def _analyze_font_statistics(self):
        if not self.parsed_lines: return
        sizes = [line.font_size for line in self.parsed_lines]
        rounded_sizes = [round(s, 1) for s in sizes]
        try:
            from statistics import mode
            self.body_font_size = mode(rounded_sizes)
        except:
            self.body_font_size = rounded_sizes[0] if rounded_sizes else 10.5

    def _tag_roles(self):
        """打标"""
        for line in self.parsed_lines:
            diff = line.font_size - self.body_font_size
            # 这里可以根据实际情况微调
            if diff > RAGConfig.HEADER_SIZE_THRESHOLD + 1.5:
                line.role = "H1"
            elif diff > RAGConfig.HEADER_SIZE_THRESHOLD:
                line.role = "H2"
            else:
                line.role = "BODY"

    def _clean_and_merge(self, lines):
        """
        优化3: 深度清洗与合并 (The Magic Function)
        解决标题断裂、页码干扰问题
        """
        cleaned_lines = []
        
        # --- 第一轮：清洗 ---
        for line in lines:
            txt = line.text.strip()
            if not txt: continue
            
            # 去除页码 (如 "- 3 -", "4", "Page 5")
            # 如果一行全是数字或只有数字和横杠，且字号接近正文，视为页码扔掉
            if re.match(r'^[-_\s0-9]+$', txt) and len(txt) < 5:
                continue
                
            # 去除 OCR 产生的奇怪单字符行
            if len(txt) == 1 and not '\u4e00' <= txt <= '\u9fa5':
                continue
                
            cleaned_lines.append(line)
            
        if not cleaned_lines: return []

        # --- 第二轮：合并同类项 ---
        merged_lines = []
        current_block = cleaned_lines[0]
        
        for i in range(1, len(cleaned_lines)):
            next_line = cleaned_lines[i]
            
            # 判断是否应该合并：
            # 1. 角色相同 (都是 H1 或 都是 H2)
            # 2. 也是正文，且上一行没有以句号/分号结束 (简单的段落拼接)
            same_role_merge = (current_block.role in ['H1', 'H2'] and next_line.role == current_block.role)
            
            # 标题必须合并，正文视情况合并
            if same_role_merge:
                current_block.text += " " + next_line.text # 合并文本
                # 字号取平均或保持最大，这里保持原样
            else:
                merged_lines.append(current_block)
                current_block = next_line
                
        merged_lines.append(current_block) # 加上最后一行
        
        return merged_lines

    def build_tree_structure(self):
        """构建树 (UI展示用)"""
        root = []
        current_h1 = None
        current_h2 = None
        
        for line in self.parsed_lines:
            # 截断过长的文本用于显示
            display_text = (line.text[:60] + '...') if len(line.text) > 60 else line.text
            
            item = {'type': line.role, 'text': display_text, 'full_text': line.text, 'page': line.page_num, 'children': []}
            
            if line.role == 'H1':
                current_h1 = item
                current_h2 = None
                root.append(current_h1)
            elif line.role == 'H2':
                current_h2 = item
                if current_h1:
                    current_h1['children'].append(current_h2)
                else:
                    root.append(current_h2)
            else: # BODY
                if current_h2:
                    current_h2['children'].append(item)
                elif current_h1:
                    current_h1['children'].append(item)
                else:
                    root.append(item)
        return root