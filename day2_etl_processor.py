import pdfplumber
import sqlite3
import json
import uuid
import os
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
class RAGConfig:
    # 文件路径配置
    PDF_PATH = "关于修订《中国国际航空股份有限公司ICS订座系统工作号管理与使用规定》的通知.pdf" # 替换为你图中的PDF文件名
    DB_PATH = "rag_production.db"
    JSON_OUTPUT_PATH = "rag_corpus_for_embedding.json"
    
    # 解析阈值 (参考 Day 1 UI 调试出的最佳参数)
    # 如果 Day 1 UI 显示正文大概是 10-12px，标题是 14px+，这里设为 2.0 比较安全
    FONT_SIZE_DIFF_THRESHOLD = 2.0  
    
    # 切片策略配置
    MAX_CHUNK_CHARS = 800       # 触发切分的阈值
    SPLIT_WINDOW_SIZE = 500     # 切分后的每段长度
    SPLIT_OVERLAP = 100         # 重叠长度

# ==========================================
# 2. 数据库管理 (SQLite Manager)
# ==========================================
class DBManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_tables()
        
    def _init_tables(self):
        # 根据你的文档[Source: 3]定义的表结构
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks_full_index (
                chunk_uuid TEXT PRIMARY KEY,
                doc_title TEXT,
                chapter_title TEXT,
                sub_title TEXT,
                full_context_text TEXT,
                pure_text TEXT,
                page_num INTEGER,
                char_count INTEGER,
                strategy_tag TEXT,
                created_at DATETIME
            )
        ''')
        self.conn.commit()
        # 清空旧数据（开发阶段方便调试，生产环境可去掉）
        self.cursor.execute('DELETE FROM chunks_full_index')
        self.conn.commit()
        
    def insert_chunk(self, data: Dict):
        self.cursor.execute('''
            INSERT INTO chunks_full_index VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['chunk_uuid'],
            data['doc_title'],
            data['chapter_title'],
            data['sub_title'],
            data['full_context_text'],
            data['pure_text'],
            data['page_num'],
            data['char_count'],
            data['strategy_tag'],
            datetime.now()
        ))
        
    def close(self):
        self.conn.commit()
        self.conn.close()

# ==========================================
# 3. 核心切片逻辑 (Smart Chunker)
# ==========================================
class SmartChunker:
    """
    负责执行你文档中[Source: 2]定义的切片策略：
    1. len <= 800: 保持原样 (Strategy: Whole)
    2. len > 800: 滑动窗口 (Strategy: Split_X_of_Y)
    """
    @staticmethod
    def process(text: str, headers: Dict, doc_title: str, page_num: int) -> List[Dict]:
        if not text.strip():
            return []
            
        chunks = []
        text_len = len(text)
        
        # 构造 Section Path 用于显示
        section_path = [doc_title]
        if headers['h1']: section_path.append(headers['h1'])
        if headers['h2']: section_path.append(headers['h2'])
        
        # --- 策略分支 ---
        
        # 策略 A: 短文本，不切分
        if text_len <= RAGConfig.MAX_CHUNK_CHARS:
            chunk_data = SmartChunker._build_packet(
                text, text, "Whole_Paragraph", 0, headers, doc_title, page_num, section_path
            )
            chunks.append(chunk_data)
            
        # 策略 B: 长文本，滑动窗口切分
        else:
            step = RAGConfig.SPLIT_WINDOW_SIZE - RAGConfig.SPLIT_OVERLAP
            # 先计算总共会切成几段
            total_segments = 0
            temp_start = 0
            while temp_start < text_len:
                total_segments += 1
                temp_start += step
            
            # 执行切分
            start = 0
            part_id = 1
            while start < text_len:
                end = min(start + RAGConfig.SPLIT_WINDOW_SIZE, text_len)
                sub_text = text[start:end]
                
                strategy_tag = f"Split_{part_id}_of_{total_segments}"
                
                chunk_data = SmartChunker._build_packet(
                    sub_text, text, strategy_tag, part_id, headers, doc_title, page_num, section_path
                )
                chunks.append(chunk_data)
                
                start += step
                part_id += 1
                
        return chunks

    @staticmethod
    def _build_packet(chunk_text, original_full_text, strategy, split_id, headers, doc_title, page, path):
        """组装符合 [Source: 2] 和 [Source: 3] 要求的 JSON 对象"""
        
        # 构造核心的 Embedding Text (算力换精度关键点)
        # 格式：Document + Chapter + Section + Content
        embedding_text = (
            f"Document: {doc_title}\n"
            f"Chapter: {headers['h1'] if headers['h1'] else 'General'}\n"
            f"Section: {headers['h2'] if headers['h2'] else ''}\n"
            f"Content: {chunk_text}"
        )
        
        # 生成唯一ID
        c_uuid = str(uuid.uuid4())
        
        # 1. 面向 SQLite 的扁平结构
        db_record = {
            "chunk_uuid": c_uuid,
            "doc_title": doc_title,
            "chapter_title": headers['h1'] or "",
            "sub_title": headers['h2'] or "",
            "full_context_text": embedding_text,
            "pure_text": chunk_text,
            "page_num": page,
            "char_count": len(chunk_text),
            "strategy_tag": strategy
        }
        
        # 2. 面向 BGE 模型的 JSON 结构
        json_record = {
            "embedding_text": embedding_text,
            "section_hint": " / ".join(path),
            "metadata": {
                "doc_title": doc_title,
                "section_id": c_uuid,
                "section_path": path,
                "page_num": page,
                "char_count": len(chunk_text),
                "strategy": strategy,
                "split_id": split_id
            },
            "original_snippet": embedding_text # 用于召回显示
        }
        
        return {"db": db_record, "json": json_record}

# ==========================================
# 4. 文档解析与状态机 (Parser & State Machine)
# ==========================================
class PDFProcessor:
    def __init__(self):
        self.db = DBManager(RAGConfig.DB_PATH)
        self.json_results = []
        
        # 状态机变量 (Context State Machine)
        self.current_h1 = None
        self.current_h2 = None
        self.buffer_text = []
        self.last_page_num = 1
        
    def run(self):
        print(f"[*] 开始处理: {RAGConfig.PDF_PATH}")
        print(f"[*] 模式: 全量上下文 (Full Context Embedding)")
        
        doc_title = os.path.basename(RAGConfig.PDF_PATH)
        
        try:
            with pdfplumber.open(RAGConfig.PDF_PATH) as pdf:
                # 1. 预扫描计算字体基准 (自动适应不同文档)
                body_font_size = self._analyze_font_stats(pdf.pages[0])
                header_threshold = body_font_size + RAGConfig.FONT_SIZE_DIFF_THRESHOLD
                print(f"[*] 自动检测: 正文约 {body_font_size:.1f}px, 标题判定阈值 > {header_threshold:.1f}px")

                # 2. 逐页解析
                for page_idx, page in enumerate(pdf.pages):
                    self._process_page(page, page_idx + 1, header_threshold, doc_title)
                    print(f"    - Page {page_idx + 1} processed.")

                # 3. 处理文档末尾残留的 buffer
                self._flush_buffer(doc_title, self.last_page_num)
                
            # 4. 导出结果
            self._export_json()
            print(f"\n[Success] 处理完成!")
            print(f"   - SQLite: {RAGConfig.DB_PATH} (已写入)")
            print(f"   - JSON:   {RAGConfig.JSON_OUTPUT_PATH} (共 {len(self.json_results)} 个切片)")
            
        except Exception as e:
            print(f"\n[Error] 处理失败: {e}")
        finally:
            self.db.close()

    def _analyze_font_stats(self, page):
        """统计页面字号众数，作为正文基准"""
        sizes = []
        for char in page.chars:
            sizes.append(char['size'])
        if not sizes: return 10.0 # fallback
        return max(set(sizes), key=sizes.count)

    def _process_page(self, page, page_num, header_threshold, doc_title):
        """单页解析逻辑：核心状态机"""
        self.last_page_num = page_num
        
        # 提取行并附带字体信息 (Simplified)
        # 实际生产中可能需要更复杂的 extract_words + grouping 逻辑，这里使用 extract_text 配合 chars 采样模拟
        # 为了代码稳健性，这里使用一种简单的行遍历策略：
        # 如果该行第一个字符很大，视为标题。
        
        # 注意：pdfplumber extract_table 等不在此处使用，我们关注纯文本流
        words = page.extract_words(extra_attrs=['size', 'fontname'])
        if not words: return

        # 简单的行重组算法
        lines = {} # y_tolerance -> text
        for w in words:
            # 按 top 位置聚类成行 (容差 3px)
            top_key = round(w['top'] / 3) * 3
            if top_key not in lines: lines[top_key] = []
            lines[top_key].append(w)
            
        sorted_y = sorted(lines.keys())
        
        for y in sorted_y:
            line_words = lines[y]
            text = " ".join([w['text'] for w in line_words])
            
            # 计算该行最大字号
            max_size = max([w['size'] for w in line_words])
            
            # --- 状态机判断逻辑 ---
            is_header = max_size >= header_threshold
            
            if is_header:
                # 关键：遇到新标题前，先结算(Flush)之前的 Buffer
                self._flush_buffer(doc_title, page_num)
                
                # 更新状态
                # 这里做一个简单的启发式：如果非常大或者是新的大章节，更新 H1，否则更新 H2
                # 实际可以根据 max_size 的梯度来分级，这里简化处理
                if max_size >= header_threshold + 2: 
                    self.current_h1 = text
                    self.current_h2 = None # 重置子标题
                else:
                    self.current_h2 = text
            else:
                # 是正文，加入缓冲区
                self.buffer_text.append(text)

    def _flush_buffer(self, doc_title, page_num):
        """将当前缓冲区的内容切片并入库"""
        if not self.buffer_text:
            return
            
        full_text = "\n".join(self.buffer_text)
        
        # 准备当前的上下文状态
        headers = {
            "h1": self.current_h1,
            "h2": self.current_h2
        }
        
        # 调用智能切片器
        packets = SmartChunker.process(full_text, headers, doc_title, page_num)
        
        for p in packets:
            # 写入 DB
            self.db.insert_chunk(p['db'])
            # 存入 JSON List
            self.json_results.append(p['json'])
            
        # 清空 Buffer
        self.buffer_text = []

    def _export_json(self):
        with open(RAGConfig.JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.json_results, f, ensure_ascii=False, indent=2)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 检查是否有文件
    if not os.path.exists(RAGConfig.PDF_PATH):
        print(f"错误: 找不到文件 {RAGConfig.PDF_PATH}")
        print("请修改代码中的 PDF_PATH 为你实际的 PDF 文件名")
    else:
        processor = PDFProcessor()
        processor.run()