# day3_backend.py
import sqlite3
import json
import requests
import urllib3
import time
import numpy as np
from datetime import datetime
from day3_config import Config

# 禁用 HTTPS 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class EmbeddingAdapter:
    """
    向量化适配器
    支持多后端切换: Intranet BGE-M3 / SiliconFlow BGE-M3 / Mock
    """
    def __init__(self, use_mock=False):
        self.use_mock = use_mock

    def get_embeddings(self, texts: list, provider_config=None, logger=None):
        """
        输入: 文本列表
        输出: 向量列表 (List[List[float]])
        provider_config: 包含 api_url, api_key, model_name 的字典
        logger: 用于回调打印日志的函数
        """
        if not texts:
            return []

        if self.use_mock:
            if logger: logger(f"[Mock] 生成 {len(texts)} 个随机向量...")
            time.sleep(0.5)
            return [np.random.rand(Config.EMBEDDING_DIM).tolist() for _ in texts]

        # 默认使用内网配置，防止 None 报错
        if not provider_config:
            provider_config = {
                "url": Config.INTRANET_API_URL,
                "key": Config.INTRANET_API_KEY,
                "model": Config.INTRANET_MODEL_NAME,
                "name": "Default"
            }

        # 真实 API 调用
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {provider_config["key"]}'
        }
        
        # 不同的提供商可能对 Payload 格式微调，但 OpenAI 格式通常通用
        payload = {
            "model": provider_config["model"],
            "input": texts,
            "encoding_format": "float" # 显式指定 float
        }

        try:
            p_name = provider_config.get('name', 'API')
            if logger: logger(f"[{p_name}] 发送请求: 批次 {len(texts)} 条...")
            start_time = time.time()
            
            # SiliconFlow 可能需要较长的超时时间
            response = requests.post(
                provider_config["url"], 
                headers=headers, 
                json=payload, 
                verify=False, # 内网需要False，硅基流动是公网通常不需要但设为False兼容性更好
                timeout=120
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result:
                    vecs = [item["embedding"] for item in result["data"]]
                    if logger: logger(f"[{p_name}] 成功 ({elapsed:.2f}s). 获得向量: {len(vecs)}")
                    return vecs
                else:
                    raise Exception(f"API 返回格式异常: {result}")
            else:
                # 尝试解析错误信息
                err_msg = response.text
                try:
                    err_json = response.json()
                    if "message" in err_json: err_msg = err_json["message"]
                except: pass
                raise Exception(f"API 错误 {response.status_code}: {err_msg}")

        except Exception as e:
            if logger: logger(f"[Error] Embedding API 调用失败: {e}")
            raise e

class DBConnector:
    """
    数据库连接器
    负责 Schema 管理与数据持久化
    """
    def __init__(self):
        self.db_path = Config.DB_PATH
        self._init_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_tables(self):
        """
        初始化数据库表结构，并自动执行 Schema 迁移
        """
        conn = self.get_connection()
        c = conn.cursor()
        
        try:
            # 1. 确保基础表存在
            c.execute('''
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
            
            # 2. [关键修复] 强制检查并添加 embedding_json 列
            c.execute("PRAGMA table_info(chunks_full_index)")
            existing_columns = [info[1] for info in c.fetchall()]
            
            if 'embedding_json' not in existing_columns:
                print(f"[DB Init] 检测到旧表结构，正在添加 'embedding_json' 列...")
                try:
                    c.execute("ALTER TABLE chunks_full_index ADD COLUMN embedding_json TEXT")
                    print("[DB Init] 列添加成功。")
                except Exception as e:
                    print(f"[DB Error] 添加列失败: {e}")
            
            conn.commit()
            
        except Exception as e:
            print(f"[DB Critical Error] 初始化失败: {e}")
        finally:
            conn.close()

    def bulk_insert(self, records):
        """批量插入数据 (使用 REPLACE INTO)"""
        if not records:
            return
            
        conn = self.get_connection()
        c = conn.cursor()
        
        try:
            for r in records:
                meta = r.get('metadata', {})
                embedding_json = json.dumps(r.get('embedding', []))
                
                # 确保字段顺序与表结构一致
                c.execute('''
                    INSERT OR REPLACE INTO chunks_full_index 
                    (chunk_uuid, doc_title, chapter_title, sub_title, full_context_text, 
                     pure_text, page_num, char_count, strategy_tag, created_at, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    meta.get('section_id', ''),         
                    meta.get('doc_title', ''),          
                    r.get('chapter_title_temp', ''),    
                    r.get('sub_title_temp', ''),        
                    r.get('embedding_text', ''),        
                    meta.get('pure_text_temp', ''),     
                    meta.get('page_num', 0),
                    meta.get('char_count', 0),
                    meta.get('strategy', 'Unknown'),
                    datetime.now(),
                    embedding_json
                ))
            conn.commit()
        except Exception as e:
            print(f"[DB Insert Error] {e}")
            raise e
        finally:
            conn.close()

    def fetch_all_vectors(self):
        """
        拉取所有向量用于仿真器内存计算
        具备极其严格的列检查，防止崩溃
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            # 1. 先检查列是否存在，如果不存在直接返回空，避免报错
            c.execute("PRAGMA table_info(chunks_full_index)")
            cols = [r[1] for r in c.fetchall()]
            if 'embedding_json' not in cols:
                print("[DB Warning] 表中缺少 embedding_json 列，无法加载向量。")
                return []

            # 2. 执行查询 (新增: 显式查询 pure_text)
            c.execute("""
                SELECT chunk_uuid, full_context_text, pure_text, embedding_json, doc_title, chapter_title, sub_title 
                FROM chunks_full_index 
                WHERE embedding_json IS NOT NULL AND embedding_json != ''
            """)
            rows = c.fetchall()
            
            results = []
            for row in rows:
                try:
                    vec_data = json.loads(row['embedding_json'])
                    if vec_data: # 确保向量非空
                        results.append({
                            'id': row['chunk_uuid'],
                            'text': row['full_context_text'],
                            'pure_text': row['pure_text'], # 核心修改: 提取纯文本
                            'vector': vec_data,
                            'doc': row['doc_title'],
                            'chapter': row['chapter_title'],
                            'sub': row['sub_title']
                        })
                except json.JSONDecodeError:
                    continue # 跳过损坏的 JSON 数据
            
            return results
            
        except sqlite3.OperationalError as e:
            print(f"[DB Fetch Error] {e}")
            return []
        finally:
            conn.close()