import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import sqlite3
import json
import uuid
import os
import queue
from datetime import datetime

# 复用 Day 1 ��解析器 (确保 pdf_structure_parser.py 在同级目录)
from pdf_structure_parser import PDFStructureParser
from config import RAGConfig as Day1Config

# ==========================================
# 1. 核心配置 (Configuration & Schema)
# ==========================================
class Day2Config:
    DB_PATH = "rag_production.db"
    JSON_OUTPUT_PATH = "rag_corpus_for_embedding.json"
    
    # 策略阈值
    MAX_CHUNK_CHARS = 800       
    SPLIT_WINDOW_SIZE = 500     
    SPLIT_OVERLAP = 100         

# ==========================================
# 2. 数据库管理 (SQLite Manager)
# ==========================================
class DBManager:
    """负责将清洗后的数据入库"""
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_tables()
        
    def _init_tables(self):
        # 扁平化存储结构，对应 Gemini 方案 V2 的表设计
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
        # 注意：这里不再执行 DELETE，以免误删 Day 3 已生成的向量数据
        # 如果需要重置，请手动删除 .db 文件或取消下面注释
        # self.cursor.execute('DELETE FROM chunks_full_index')
        # self.conn.commit()
        
    def insert_chunk(self, data):
        # [修复] 显式指定列名，解决 "table has 11 columns but 10 values were supplied" 问题
        # 无论数据库后续增加了 embedding_json 还是其他字段，这里只插入 Day 2 负责的 10 个基础字段
        self.cursor.execute('''
            INSERT INTO chunks_full_index 
            (chunk_uuid, doc_title, chapter_title, sub_title, full_context_text, 
             pure_text, page_num, char_count, strategy_tag, created_at)
            VALUES 
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
        
    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

# ==========================================
# 3. 智能切片逻辑 (Smart Chunker)
# ==========================================
class SmartChunker:
    """
    ✨ 方案 2 修复版本：Day 2 的切片逻辑
    核心改进：
    1. 在 JSON 记录中显式包含 pure_text 字段
    2. 确保 pure_text 包含完整的段落内容（不含标题）
    3. embedding_text 包含完整的标题+内容
    """
    
    @staticmethod
    def process_paragraph(doc_title, h1, h2, paragraph_text, page_num):
        """
        输入：文档名, 当前H1, 当前H2, 正文段落, 页码
        输出：一个列表，包含1个或多个切片字典 (DB格式 + JSON格式)
        
        ✨ 核心修复：
        - paragraph_text 是原始的、完整的段落文本
        - 我们在此处就明确分离出 pure_text 和 embedding_text
        - 确保 pure_text 在整个流转过程中不会丢失
        """
        results = []
        text_len = len(paragraph_text)
        
        # 构造路径供 section_hint 使用
        section_path_list = [t for t in [doc_title, h1, h2] if t]
        section_path_str = " / ".join(section_path_list)
        
        # --- 策略分支 ---
        chunks_to_create = []
        
        if text_len <= Day2Config.MAX_CHUNK_CHARS:
            # 策略 A: 保持原样 (Whole)
            chunks_to_create.append({
                "text": paragraph_text,
                "strategy": "Whole_Paragraph",
                "split_id": 0
            })
        else:
            # 策略 B: 滑动窗口切分 (Split)
            start = 0
            part_id = 1
            step = Day2Config.SPLIT_WINDOW_SIZE - Day2Config.SPLIT_OVERLAP
            
            # 计算总共会切成几段
            total_parts = 0
            temp_start = 0
            while temp_start < text_len:
                total_parts += 1
                temp_start += step
            
            while start < text_len:
                end = min(start + Day2Config.SPLIT_WINDOW_SIZE, text_len)
                sub_text = paragraph_text[start:end]
                
                chunks_to_create.append({
                    "text": sub_text,
                    "strategy": f"Split_{part_id}_of_{total_parts}",
                    "split_id": part_id
                })
                
                if end == text_len: 
                    break
                start += step
                part_id += 1

        # --- 组装数据对象 ---
        for item in chunks_to_create:
            pure_text = item['text']  # ✨ 直接使用切片后的纯文本
            
            # 核心：全量上下文融合
            # 格式：Document -> Chapter -> Section -> Content
            embedding_text = (
                f"Document: {doc_title}\n"
                f"Chapter: {h1 if h1 else 'General'}\n"
                f"Section: {h2 if h2 else 'Intro'}\n"
                f"Content: {pure_text}"
            )
            
            chunk_uuid = str(uuid.uuid4())
            
            # 1. DB 记录格式 (扁平)
            db_record = {
                "chunk_uuid": chunk_uuid,
                "doc_title": doc_title,
                "chapter_title": h1 or "",
                "sub_title": h2 or "",
                "full_context_text": embedding_text,
                "pure_text": pure_text,  # ✨ 保证完整
                "page_num": page_num,
                "char_count": len(pure_text),
                "strategy_tag": item['strategy']
            }
            
            # 2. JSON 记录格式 (嵌套，适配 BGE + Day 3)
            # ✨ 关键修复：在 JSON 中显式包含 pure_text 字段
            json_record = {
                "embedding_text": embedding_text,
                "pure_text": pure_text,  # ✨ 新增：直接在 JSON 中保存 pure_text
                "section_hint": section_path_str,
                "metadata": {
                    "doc_title": doc_title,
                    "section_id": chunk_uuid,
                    "section_path": section_path_list,
                    "page_num": page_num,
                    "char_count": len(pure_text),
                    "strategy": item['strategy'],
                    "split_id": item['split_id'],
                    "pure_text": pure_text  # ✨ 也在 metadata 中备份
                },
                "original_snippet": section_path_str  # ✨ 简化为路径字符串
            }
            
            results.append({"db": db_record, "json": json_record})
            
        return results

# ==========================================
# 4. ETL 核心流水线 (Worker)
# ==========================================
class ETLWorker(threading.Thread):
    def __init__(self, filepath, use_ocr, message_queue, result_callback):
        super().__init__()
        self.filepath = filepath
        self.use_ocr = use_ocr
        self.msg_q = message_queue
        self.callback = result_callback
        self.stop_event = threading.Event()

    # 适配 Day 1 Parser 的回调接口
    class ProgressSignalAdapter:
        def __init__(self, queue_obj):
            self.q = queue_obj
        def emit(self, msg, val):
            self.q.put(("LOG", f"[Day1 Parser] {msg}"))

    def run(self):
        try:
            self.msg_q.put(("LOG", "=== 阶段 1: 启动文档结构解析 (Day 1 Core) ==="))
            doc_title = os.path.basename(self.filepath)
            
            # 1. 调用 Day 1 解析器提取结构化 Lines
            parser = PDFStructureParser(self.filepath, use_ocr=self.use_ocr)
            # 使用适配器将 parser 的 PyQt 信号转为 Queue 消息
            signal_adapter = self.ProgressSignalAdapter(self.msg_q)
            
            # 执行解析 (这里复用了 Day 1 强大的清洗逻辑)
            parsed_blocks = parser.parse(callback_signal=signal_adapter)
            
            self.msg_q.put(("LOG", f"结构提取完成，共获取 {len(parsed_blocks)} 个文本块"))
            self.msg_q.put(("LOG", f"检测到正文基准字号: {parser.body_font_size}"))

            # 2. 状态机与组装 (Day 2 Core)
            self.msg_q.put(("LOG", "=== 阶段 2: 上下文锚点融合与切片 ==="))
            
            db_manager = DBManager(Day2Config.DB_PATH)
            json_output = []
            
            current_h1 = None
            current_h2 = None
            total_chunks = 0
            
            # ✨ 新增：数据质量统计
            total_input_chars = 0
            total_output_chars = 0
            
            # 状态机循环
            for i, block in enumerate(parsed_blocks):
                # 更新上下文状态
                if block.role == 'H1':
                    current_h1 = block.text
                    current_h2 = None # 切换章节时重置子标题
                    self.msg_q.put(("LOG", f">> 锁定一级标题: {current_h1[:30]}..."))
                    continue
                elif block.role == 'H2':
                    current_h2 = block.text
                    self.msg_q.put(("LOG", f"  > 锁定二级标题: {current_h2[:30]}..."))
                    continue
                
                # 处理正文 (BODY)
                if block.role == 'BODY':
                    total_input_chars += len(block.text)
                    
                    # 调用智能切分器
                    packets = SmartChunker.process_paragraph(
                        doc_title=doc_title,
                        h1=current_h1,
                        h2=current_h2,
                        paragraph_text=block.text,
                        page_num=block.page_num
                    )
                    
                    for p in packets:
                        db_manager.insert_chunk(p['db'])
                        json_output.append(p['json'])
                        total_chunks += 1
                        total_output_chars += len(p['db']['pure_text'])
                        
                        # 实时发送前几个切片给 GUI 做"切片显微镜"展示
                        if total_chunks <= 5 or total_chunks % 10 == 0:
                            self.msg_q.put(("PREVIEW", p['json']))

            # 3. 收尾
            db_manager.commit()
            db_manager.close()
            
            # 导出 JSON
            with open(Day2Config.JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, ensure_ascii=False, indent=2)
                
            self.msg_q.put(("LOG", "="*50))
            self.msg_q.put(("LOG", f"[SUCCESS] ETL 完成!"))
            self.msg_q.put(("LOG", f"总输入块数: {len(parsed_blocks)}"))
            self.msg_q.put(("LOG", f"总输出切片: {total_chunks}"))
            self.msg_q.put(("LOG", f"输入总字数: {total_input_chars}"))
            self.msg_q.put(("LOG", f"输出总字数: {total_output_chars}"))
            self.msg_q.put(("LOG", f"数据完整率: {total_output_chars/max(total_input_chars, 1)*100:.1f}%"))
            self.msg_q.put(("LOG", f"数据库: {Day2Config.DB_PATH}"))
            self.msg_q.put(("LOG", f"语料JSON: {Day2Config.JSON_OUTPUT_PATH}"))
            self.msg_q.put(("LOG", "="*50))
            
            self.callback(True)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.msg_q.put(("LOG", f"[ERROR] {str(e)}"))
            self.msg_q.put(("LOG", err_msg))
            self.callback(False)

# ==========================================
# 5. GUI 主界面 (Chunk Inspector)
# ==========================================
class Day2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Day 2: 结构化组装与切片显微镜 (Gemini V2 - Method 2 Enhanced)")
        self.root.geometry("1100x700")
        
        # 消息队列用于线程通信
        self.msg_queue = queue.Queue()
        
        self._init_ui()
        self._check_queue() # 启动定时器轮询消息

    def _init_ui(self):
        # --- 顶部控制栏 ---
        top_frame = tk.Frame(self.root, pady=10, padx=10, bg="#f0f0f0")
        top_frame.pack(fill="x")
        
        tk.Label(top_frame, text="PDF 源文件:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(side="left")
        self.path_entry = tk.Entry(top_frame, width=40)
        self.path_entry.pack(side="left", padx=5)
        tk.Button(top_frame, text="浏览...", command=self.browse_file).pack(side="left")

        # 默认使用 Day 1 的��置逻辑，默认开启 OCR
        self.ocr_var = tk.BooleanVar(value=True)
        self.ocr_check = tk.Checkbutton(top_frame, text="启用 Tesseract OCR (继承 Day 1)", variable=self.ocr_var, bg="#f0f0f0")
        self.ocr_check.pack(side="left", padx=10)

        self.btn_run = tk.Button(top_frame, text="▶ 开始 ETL 流水线", bg="#007ACC", fg="white", 
                                font=("Arial", 11, "bold"), command=self.start_etl)
        self.btn_run.pack(side="left", padx=10)

        # --- 主体分割 (PanedWindow) ---
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # 左侧：日志控制台
        left_frame = tk.LabelFrame(paned, text="系统日志 (System Log)", padx=5, pady=5)
        paned.add(left_frame, minsize=400)
        
        self.console = scrolledtext.ScrolledText(left_frame, state="disabled", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.console.pack(fill="both", expand=True)

        # 右侧：切片显微镜 (Chunk Inspector) 
        right_frame = tk.LabelFrame(paned, text="切片显微镜 (Chunk Inspector - Preview)", padx=5, pady=5)
        paned.add(right_frame, minsize=500)
        
        # 预览树形结构或文本
        self.preview_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, bg="#ffffee", font=("Segoe UI", 10))
        self.preview_text.pack(fill="both", expand=True)
        # 添加标签说明
        info_lbl = tk.Label(right_frame, text="✨ Method 2 Enhanced: 此处显示 pure_text 字段，验证数据完整性", fg="blue", bg="#ffffee", font=("Segoe UI", 9, "bold"))
        info_lbl.pack(side="bottom", fill="x")

    def browse_file(self):
        fn = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if fn: 
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, fn)

    def log(self, msg):
        self.console.configure(state="normal")
        self.console.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.console.see(tk.END)
        self.console.configure(state="disabled")

    def show_preview(self, json_data):
        """
        ✨ 增强的预览显示：重点展示 pure_text 字段
        """
        self.preview_text.insert(tk.END, f"\n{'='*60} 新切片生成 {'='*60}\n", "header")
        
        # 关键字段高亮显示
        self.preview_text.insert(tk.END, f"UUID: {json_data.get('metadata', {}).get('section_id', 'N/A')[:12]}...\n", "info")
        self.preview_text.insert(tk.END, f"策略: {json_data.get('metadata', {}).get('strategy', 'N/A')}\n", "info")
        self.preview_text.insert(tk.END, f"字数: {json_data.get('metadata', {}).get('char_count', 0)}\n", "info")
        
        # ✨ 重点展示 pure_text
        pure_text = json_data.get('pure_text', '')
        self.preview_text.insert(tk.END, f"\n[pure_text (字数: {len(pure_text)})]:\n", "highlight")
        self.preview_text.insert(tk.END, f"{pure_text[:200]}{'...' if len(pure_text) > 200 else ''}\n", "body")
        
        # embedding_text 作为参考
        embedding_text = json_data.get('embedding_text', '')
        self.preview_text.insert(tk.END, f"\n[embedding_text (字数: {len(embedding_text)})]:\n", "highlight")
        self.preview_text.insert(tk.END, f"{embedding_text[:200]}{'...' if len(embedding_text) > 200 else ''}\n", "body")
        
        self.preview_text.insert(tk.END, "-"*60 + "\n\n")
        self.preview_text.see(tk.END)

    def start_etl(self):
        path = self.path_entry.get()
        if not os.path.exists(path):
            messagebox.showerror("错误", "文件路径无效")
            return
            
        self.console.configure(state="normal"); self.console.delete(1.0, tk.END); self.console.configure(state="disabled")
        self.preview_text.delete(1.0, tk.END)
        self.btn_run.config(state="disabled", text="正在运行...")
        
        # 启动后台线程
        worker = ETLWorker(path, self.ocr_var.get(), self.msg_queue, self.on_finished)
        worker.start()

    def on_finished(self, success):
        self.btn_run.config(state="normal", text="▶ 开始 ETL 流水线")
        if success:
            messagebox.showinfo("完成", f"✨ ETL 处理成功！\n数据已写入 {Day2Config.DB_PATH}\n\nJSON 中已包含 pure_text 字段，Day 3 无需再做字符串分割")
        else:
            messagebox.showerror("失败", "ETL 处理过程中发生错误，请查看日志。")

    def _check_queue(self):
        """轮询消息队列更新 UI"""
        try:
            while True:
                msg_type, content = self.msg_queue.get_nowait()
                if msg_type == "LOG":
                    self.log(content)
                elif msg_type == "PREVIEW":
                    self.show_preview(content)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = Day2GUI(root)
    root.mainloop()