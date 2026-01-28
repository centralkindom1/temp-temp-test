# day3_rag_simulator.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import threading
import queue
import numpy as np
import concurrent.futures
from datetime import datetime
from day3_config import Config
from day3_backend import EmbeddingAdapter, DBConnector

class RAGSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Day 3: RAG 仿真器 & 向量仓库 (双通道版: Intranet/SiliconFlow)")
        self.root.geometry("1100x950")
        
        # 线程通信队列
        self.msg_queue = queue.Queue()
        
        # === 后端组件初始化 ===
        # DBConnector 是我们的"仓库管理员"，负责连接 rag_production.db
        self.db_conn = DBConnector()
        # Adapter 是我们的"翻译官"，负责调用 API
        self.adapter = EmbeddingAdapter(use_mock=False) 
        
        # 仿真器内存：从 DB 加载的向量和元数据将缓存在这里
        self.memory_vectors = []
        
        self._init_ui()
        
        # 启动队列轮询
        self.root.after(100, self._check_queue)
        
        # === 启动时自动尝试挂载默认路径 ===
        # 虽然增加了手动挂载，但为了方便，启动时还是自动挂载一次
        self.reload_memory_db()
        
    def _init_ui(self):
        main_layout = tk.Frame(self.root)
        main_layout.pack(fill="both", expand=True)

        # ==========================================
        # 区域 1：工序 A - 向量化入库 (The Truck)
        # ==========================================
        ingest_frame = tk.LabelFrame(main_layout, text="工序 A: 向量化入库 (Source: JSON -> Target: DB)", padx=10, pady=10)
        ingest_frame.pack(fill="x", padx=10, pady=5)
        
        # 第一行：文件选择 (JSON 源)
        file_box = tk.Frame(ingest_frame)
        file_box.pack(fill="x", pady=2)
        tk.Label(file_box, text="[运货卡车] Day2 产物 JSON:").pack(side="left")
        self.json_path_entry = tk.Entry(file_box, width=50)
        self.json_path_entry.pack(side="left", padx=5)
        self.json_path_entry.insert(0, Config.INPUT_JSON_PATH)
        tk.Button(file_box, text="浏览...", command=self.browse_json_file).pack(side="left")

        # 第二行：API 配置与并发控制
        config_box = tk.Frame(ingest_frame)
        config_box.pack(fill="x", pady=8)
        
        # 1. API 选择
        tk.Label(config_box, text="API 提供商:", font=("bold")).pack(side="left")
        self.provider_var = tk.StringVar(value="Intranet (AirChina)")
        self.provider_combo = ttk.Combobox(config_box, textvariable=self.provider_var, state="readonly", width=22)
        self.provider_combo['values'] = ("Intranet (AirChina)", "SiliconFlow (Public)")
        self.provider_combo.pack(side="left", padx=5)
        
        # 2. 批次大小
        tk.Label(config_box, text=" |  Batch Size:").pack(side="left", padx=2)
        self.batch_size_spin = tk.Spinbox(config_box, from_=1, to=50, width=5)
        self.batch_size_spin.delete(0, "end")
        self.batch_size_spin.insert(0, Config.DEFAULT_BATCH_SIZE)
        self.batch_size_spin.pack(side="left")
        
        # 3. 最大并发
        tk.Label(config_box, text=" |  Max Concurrency:").pack(side="left", padx=2)
        self.concurrency_spin = tk.Spinbox(config_box, from_=1, to=10, width=5)
        self.concurrency_spin.delete(0, "end")
        self.concurrency_spin.insert(0, Config.DEFAULT_CONCURRENCY)
        self.concurrency_spin.pack(side="left")

        # 4. 启动按钮
        self.btn_ingest = tk.Button(config_box, text="🚀 启动批量向量化入库", bg="#007ACC", fg="white", font=("Arial", 10, "bold"), command=self.start_ingestion_thread)
        self.btn_ingest.pack(side="left", padx=20)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(ingest_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        # ==========================================
        # 区域 2：工序 B - RAG 仿真验证 (The Warehouse)
        # ==========================================
        sim_frame = tk.LabelFrame(main_layout, text="工序 B: RAG 仿真验证 (Source: DB Warehouse)", padx=10, pady=10)
        sim_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # --- 新增：数据库手动挂载控制区 ---
        db_control_box = tk.Frame(sim_frame, bg="#f0f0f0", bd=1, relief="groove")
        db_control_box.pack(fill="x", pady=5, padx=5)
        
        tk.Label(db_control_box, text="[稳固仓库] DB文件路径:", bg="#f0f0f0").pack(side="left", padx=5)
        self.db_path_entry = tk.Entry(db_control_box, width=50)
        self.db_path_entry.pack(side="left", padx=5, pady=5)
        self.db_path_entry.insert(0, Config.DB_PATH) # 默认填入 Config 里的路径
        
        tk.Button(db_control_box, text="📂 选择仓库...", command=self.browse_db_file).pack(side="left", padx=2)
        tk.Button(db_control_box, text="🔄 立即挂载/刷新", bg="#ffc107", command=self.reload_memory_db).pack(side="left", padx=10)
        
        # 状态显示标签
        self.db_status_label = tk.Label(db_control_box, text="状态: 等待挂载...", bg="#f0f0f0", fg="#666666", font=("Consolas", 9, "bold"))
        self.db_status_label.pack(side="left", padx=10)
        
        # --- 搜索区域 ---
        search_box = tk.Frame(sim_frame)
        search_box.pack(fill="x", pady=10)
        tk.Label(search_box, text="输入测试问题:", font=("Arial", 12, "bold")).pack(side="left")
        self.query_entry = tk.Entry(search_box, font=("Arial", 12))
        self.query_entry.pack(side="left", fill="x", expand=True, padx=10)
        self.query_entry.bind("<Return>", lambda event: self.run_simulation())
        
        btn_search = tk.Button(search_box, text="🔍 计算相似度召回", bg="#28a745", fg="white", font=("Arial", 11, "bold"), command=self.run_simulation)
        btn_search.pack(side="left")

        # 结果显示区
        self.result_area = scrolledtext.ScrolledText(sim_frame, font=("Segoe UI", 10), height=15)
        self.result_area.pack(fill="both", expand=True)
        
        # 样式配置
        self.result_area.tag_config("title_hit", background="yellow", foreground="black", font=("Segoe UI", 10, "bold"))
        self.result_area.tag_config("score", foreground="red", font=("Segoe UI", 10, "bold"))
        self.result_area.tag_config("meta", foreground="#666666", font=("Consolas", 9))
        self.result_area.tag_config("source_db", foreground="blue", font=("Consolas", 8, "italic"))
        self.result_area.tag_config("pure_body", foreground="black", font=("Segoe UI", 10))

        # ==========================================
        # 区域 3：实时控制台日志
        # ==========================================
        console_frame = tk.LabelFrame(main_layout, text="后台通讯日志 (Console Log)", padx=10, pady=5, bg="#1e1e1e", fg="white")
        console_frame.pack(fill="x", padx=10, pady=5, side="bottom")
        
        self.console_area = scrolledtext.ScrolledText(console_frame, height=8, bg="black", fg="#00FF00", font=("Consolas", 9))
        self.console_area.pack(fill="both", expand=True)

    def log(self, msg):
        """线程安全的日志发送方法"""
        self.msg_queue.put(("LOG", msg))

    def _check_queue(self):
        """主线程轮询队列，更新 GUI"""
        try:
            while True:
                msg_type, content = self.msg_queue.get_nowait()
                
                if msg_type == "LOG":
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.console_area.insert(tk.END, f"[{timestamp}] {content}\n")
                    self.console_area.see(tk.END)
                
                elif msg_type == "PROGRESS":
                    self.progress_bar['value'] = content
                
                elif msg_type == "STATUS_DONE":
                    messagebox.showinfo("完成", content)
                    self.btn_ingest.config(state="normal")
                    # 入库完成后，自动刷新
                    self.reload_memory_db() 
                
                elif msg_type == "ERROR":
                    messagebox.showerror("错误", content)
                    self.btn_ingest.config(state="normal")
                
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_queue)

    # --- 文件选择辅助 ---
    def browse_json_file(self):
        fn = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if fn: 
            self.json_path_entry.delete(0, tk.END)
            self.json_path_entry.insert(0, fn)

    def browse_db_file(self):
        fn = filedialog.askopenfilename(filetypes=[("SQLite DB", "*.db"), ("All Files", "*.*")])
        if fn:
            self.db_path_entry.delete(0, tk.END)
            self.db_path_entry.insert(0, fn)
            # 选完文件后，自动触发一次挂载，提升体验
            self.reload_memory_db()

    def get_current_api_config(self):
        """根据下拉框选择获取配置"""
        choice = self.provider_var.get()
        if "SiliconFlow" in choice:
            return {
                "name": "SiliconFlow",
                "url": Config.SILICON_API_URL,
                "key": Config.SILICON_API_KEY,
                "model": Config.SILICON_MODEL_NAME
            }
        else:
            return {
                "name": "Intranet",
                "url": Config.INTRANET_API_URL,
                "key": Config.INTRANET_API_KEY,
                "model": Config.INTRANET_MODEL_NAME
            }

    # --- 核心逻辑：从数据库(仓库)加载数据 ---
    def reload_memory_db(self):
        """
        ✨ 方案 2 修复版本：连接 DB，拉取 embedding_json, full_context_text 和 pure_text 到内存。
        支持从 UI 输入框动态读取 DB 路径。
        增强的数据验证和修复能力。
        """
        # 1. 获取界面上配置的 DB 路径
        target_db_path = self.db_path_entry.get().strip()
        if not target_db_path:
            target_db_path = Config.DB_PATH # 回退到默认
        
        self.log(f"正在尝试挂载数据库: {target_db_path} ...")
        
        # 2. 更新 Connector 的路径
        # 注意：这里我们动态修改 db_conn 实例的路径属性，以便后续操作都针对新 DB
        self.db_conn.db_path = target_db_path
        
        if not os.path.exists(target_db_path):
            self.log(f"错误: 找不到文件 {target_db_path}")
            self.db_status_label.config(text=f"状态: 文件不存在", fg="red")
            self.memory_vectors = []
            return

        # 3. 调用 Backend 的方法
        # 此时 db_conn.fetch_all_vectors() 内部会使用 self.db_path (也就是我们刚才设置的 target_db_path)
        raw_data = self.db_conn.fetch_all_vectors()
        
        if not raw_data:
            self.log("挂载成功，但数据库为空 (没有有效向量)。")
            self.memory_vectors = []
            self.db_status_label.config(text=f"状态: 空数据库 | Path: {os.path.basename(target_db_path)}", fg="#ff8800")
            return

        # 4. 转换数据格式 + 增强的数据完整性检查
        self.memory_vectors = []
        skip_count = 0
        
        for item in raw_data:
            try:
                # ✨ 额外的数据完整性检查
                if not item.get('pure_text') or not item['pure_text'].strip():
                    self.log(f"⚠️ 警告：记录 {item['id'][:8]}... 缺少有效的 pure_text，已跳过")
                    skip_count += 1
                    continue
                
                item['np_vector'] = np.array(item['vector'])
                self.memory_vectors.append(item)
            except Exception as e:
                self.log(f"⚠️ 警告：处理记录 {item.get('id', 'unknown')[:8]}... 时出错: {e}")
                skip_count += 1
                continue
            
        count = len(self.memory_vectors)
        if skip_count > 0:
            self.log(f"[Info] 数据库挂载成功！已跳过 {skip_count} 条损坏记录。")
        self.log(f"[Success] 内存索引已构建，共 {count} 条有效数据可用。")
        self.db_status_label.config(text=f"状态: 已挂载 ✅ | 索引量: {count} 条", fg="green")

    # --- 线程工作逻辑：入库 (JSON -> API -> DB) ---
    def start_ingestion_thread(self):
        path = self.json_path_entry.get()
        if not os.path.exists(path):
            messagebox.showerror("错误", "找不到输入的 JSON 文件")
            return
        
        try:
            batch_size = int(self.batch_size_spin.get())
            max_workers = int(self.concurrency_spin.get())
            if batch_size < 1 or max_workers < 1: raise ValueError
        except:
            messagebox.showerror("错误", "批次大小或并发数必须为正整数")
            return

        api_config = self.get_current_api_config()
        
        self.btn_ingest.config(state="disabled")
        self.log(f"启动入库任务 | 源: JSON | 目标: DB | 并发: {max_workers}")
        
        threading.Thread(
            target=self.run_ingestion, 
            args=(path, api_config, batch_size, max_workers), 
            daemon=True
        ).start()

    def run_ingestion(self, json_path, api_config, batch_size, max_workers):
        """
        ✨ 方案 2 修复版本：批量入库逻辑
        核心改进：在处理 batch 时，优先从 JSON 的 pure_text 读取，实现多级降级策略
        """
        try:
            self.log("正在解析 JSON 文件...")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_items = len(data)
            self.log(f"解析成功，共 {total_items} 条数据待处理。")
            
            # 数据检查：扫描 JSON 中是否包含 pure_text 字段
            sample_item = data[0] if data else {}
            has_pure_text = 'pure_text' in sample_item
            self.log(f"[Info] JSON 数据结构检查：pure_text 字段 {'✅ 已包含' if has_pure_text else '❌ 缺失'}")
            
            batches = []
            for i in range(0, total_items, batch_size):
                batch_data = data[i:i+batch_size]
                batch_texts = [item['embedding_text'] for item in batch_data]
                batches.append({
                    'index': i,
                    'data': batch_data,
                    'texts': batch_texts
                })
            
            processed_data = []
            processed_count = 0
            lock = threading.Lock()
            
            def process_batch(batch_info):
                """
                ✨ 核心处理函数：实现方案 2 的多级降级策略
                """
                texts = batch_info['texts']
                def thread_logger(msg):
                    if "Error" in msg or "error" in msg: 
                        self.log(msg)
                
                try:
                    vectors = self.adapter.get_embeddings(texts, provider_config=api_config, logger=thread_logger)
                    
                    result_records = []
                    for idx, item in enumerate(batch_info['data']):
                        meta = item.get('metadata', {})
                        path_list = meta.get('section_path', [])
                        h1 = path_list[1] if len(path_list) > 1 else ""
                        h2 = path_list[2] if len(path_list) > 2 else ""
                        
                        record = item.copy()
                        record['embedding'] = vectors[idx]
                        record['chapter_title_temp'] = h1
                        record['sub_title_temp'] = h2
                        
                        # ✨ 方案 2 的核心修复：多级降级策略获取 pure_text
                        pure_text = ""
                        
                        # 第一优先级：直接从 JSON 的 pure_text 字段读取
                        if 'pure_text' in item and item['pure_text']:
                            pure_text = item['pure_text'].strip()
                            
                        # 第二优先级：从 metadata 中读取
                        elif 'pure_text' in meta and meta['pure_text']:
                            pure_text = meta['pure_text'].strip()
                        
                        # 第三优先级：从 embedding_text 分割提取
                        else:
                            embedding_text = item.get('embedding_text', '')
                            if "Content: " in embedding_text:
                                pure_text = embedding_text.split("Content: ", 1)[1].strip()
                            else:
                                pure_text = embedding_text.strip()
                        
                        # 最后保底：确保 pure_text 不为空
                        if not pure_text:
                            pure_text = item.get('embedding_text', '').strip()
                        
                        # 将处理后的 pure_text 保存到 metadata 和 record，供后续 bulk_insert 使用
                        meta['pure_text'] = pure_text
                        record['metadata'] = meta
                        
                        result_records.append(record)
                    return result_records
                except Exception as e:
                    self.log(f"[Batch Error] 索引 {batch_info['index']} 失败: {e}")
                    return None

            self.log(f"开始并发处理，线程池大小: {max_workers}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(process_batch, b): b for b in batches}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    results = future.result()
                    if results:
                        with lock:
                            # 这里调用 backend 的 bulk_insert，数据真正存入 Warehouse (DB)
                            # bulk_insert 内部已经集成了方案 2 的逻辑
                            self.db_conn.bulk_insert(results)
                            processed_data.extend(results)
                            processed_count += len(results)
                            
                            progress = (processed_count / total_items) * 100
                            self.msg_queue.put(("PROGRESS", progress))
                            
                            if processed_count % (batch_size * 2) == 0:
                                self.log(f"进度: {processed_count}/{total_items} 已入库")
            
            self.log("="*50)
            self.log("入库任务全部完成！数据已安全存入数据库。")
            self.log(f"总处理数: {len(processed_data)} | 成功率: {len(processed_data)/total_items*100:.1f}%")
            self.log("="*50)
            self.msg_queue.put(("STATUS_DONE", f"入库成功！共 {len(processed_data)} 条数据。\n已存入 DB，ready for RAG simulation."))
            
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.log(f"FATAL ERROR: {str(e)}")
            print(err)
            self.msg_queue.put(("ERROR", f"���理异常: {str(e)}"))

    # --- 仿真搜索逻辑 (Read from DB Memory) ---
    def run_simulation(self):
        """
        ✨ 方案 2 修复版本：RAG 仿真搜索
        使用从数据库加载的、经过验证的 pure_text 和 full_context_text 进行召回
        """
        query = self.query_entry.get()
        if not query: return
        
        # 强制检查：必须基于数据库内容
        if not self.memory_vectors:
            messagebox.showwarning("警告", "当前未挂载数据库或数据库为空。\n请检查文件路径并点击'立即挂载/刷新'")
            return

        self.result_area.configure(state="normal")
        self.result_area.delete(1.0, tk.END)
        
        api_config = self.get_current_api_config()
        self.log(f"正在向量化问题: '{query}' ...")
        
        try:
            q_vec = self.adapter.get_embeddings([query], provider_config=api_config, logger=None)[0]
            q_vec_np = np.array(q_vec)
        except Exception as e:
            self.result_area.insert(tk.END, f"[Error] 向量化失败: {e}\n")
            self.log(f"向量化失败: {e}")
            return

        self.log(f"正在 {len(self.memory_vectors)} 条数据中检索 (余弦相似度)...")
        scores = []
        q_norm = np.linalg.norm(q_vec_np)
        
        # 这里的 item 来源于 reload_memory_db 中拉取的 DB 数据
        for item in self.memory_vectors:
            d_vec_np = item['np_vector']
            d_norm = np.linalg.norm(d_vec_np)
            
            if q_norm == 0 or d_norm == 0:
                sim = 0
            else:
                sim = np.dot(q_vec_np, d_vec_np) / (q_norm * d_norm)
            
            scores.append((sim, item))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:3]

        self.result_area.insert(tk.END, f"\n{'='*20} 仿真召回结果 (Top 3) {'='*20}\n")
        
        current_db_name = os.path.basename(self.db_path_entry.get())
        self.result_area.insert(tk.END, f"数据源: {current_db_name} (Pure Text Fusion - Method 2 Enhanced)\n\n", "source_db")
        
        if not top_k:
            self.result_area.insert(tk.END, "无匹配结果。\n")

        for i, (score, item) in enumerate(top_k):
            self.log(f"Top {i+1} Score: {score:.4f} | Doc: {item['doc']}")
            
            self.result_area.insert(tk.END, f"Rank {i+1} | ")
            self.result_area.insert(tk.END, f"相似度: {score:.4f}\n", "score")
            
            # 1. 标题
            title_text = f"文档: {item['doc']} >> 章: {item['chapter']} >> 节: {item['sub']}\n"
            self.result_area.insert(tk.END, title_text, "title_hit")
            
            # 2. 融合内容展示 (Full Context Header + Pure Text Body)
            # ✨ 方案 2 修复：pure_text 现在是直接从数据库读取的、经过验证的值
            full_context = item['text']
            pure_text_body = item.get('pure_text', "")
            
            # 提取 Header (包含路径信息的部分)
            header_text = ""
            if "Content: " in full_context:
                header_text = full_context.split("Content: ", 1)[0] + "Content: "
            else:
                header_text = "Metadata Header (Parse Failed):"

            # 显示 Header
            self.result_area.insert(tk.END, f"[Full Context Header]:\n{header_text}\n", "meta")
            
            # 显示 Body (优先使用经过验证的 Pure Text)
            if pure_text_body and len(pure_text_body.strip()) > 0:
                 self.result_area.insert(tk.END, f"[Pure Text Body - DB Source]:\n{pure_text_body.strip()}\n", "pure_body")
            elif "Content: " in full_context:
                 # Fallback (不应该到这里，如果数据库加载正常)
                 fallback_body = full_context.split("Content: ", 1)[1]
                 self.result_area.insert(tk.END, f"[Body (Fallback)]:\n{fallback_body.strip()}\n")
            else:
                 self.result_area.insert(tk.END, f"[Body]:\n{full_context}\n")

            self.result_area.insert(tk.END, "-"*50 + "\n")

        self.result_area.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGSimulatorGUI(root)
    root.mainloop()