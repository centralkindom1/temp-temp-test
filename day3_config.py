# day3_config.py
import os

class Config:
    # === 默认配置 (内网) ===
    INTRANET_API_URL = "https://aiplus.airchina.com.cn:18080/v1/embeddings"
    INTRANET_API_KEY = "sk-fXM4W0CdcK"
    INTRANET_MODEL_NAME = "bge-m3"

    # === 硅基流动配置 (SiliconFlow) ===
    # 来源: 用户指定 (硬编码)
    SILICON_API_URL = "https://api.siliconflow.cn/v1/embeddings"
    # 使用最新的 Key
    SILICON_API_KEY = "sk-udyhequsmccvaccqywfmkksezoeuvfvgbmtpudhzbrdeexac"
    # 硅基流动的 BGE-M3 模型名称
    SILICON_MODEL_NAME = "BAAI/bge-m3"
    
    # === 路径配置 ===
    # Day 2 生成的输入文件
    INPUT_JSON_PATH = "rag_corpus_for_embedding.json"
    # Day 3 输出的最终数据库
    DB_PATH = "rag_production.db"
    # Day 3 备份的含向量JSON
    FINAL_JSON_PATH = "final_embedding_corpus.json"
    
    # === 向量化默认配置 ===
    DEFAULT_BATCH_SIZE = 8 # 默认批处理大小
    DEFAULT_CONCURRENCY = 2 # 默认并发数

    EMBEDDING_DIM = 1024 # BGE-M3 维度
