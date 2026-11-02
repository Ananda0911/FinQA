"""全局配置"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COMPANIES_DIR = DATA_DIR / "companies"      # 每个公司一个子目录（索引/切片/页面）
GRAPH_DIR = DATA_DIR / "graph"              # GraphRAG 图谱数据
CACHE_DIR = DATA_DIR / "cache"              # 表格序列化等缓存
KNOWLEDGE_FILE = DATA_DIR / "knowledge.jsonl"  # 对话沉淀知识

# ---------- DashScope ----------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LLM_MODEL = os.getenv("FINQA_LLM_MODEL", "qwen-plus")
LLM_MODEL_FAST = os.getenv("FINQA_LLM_MODEL_FAST", "qwen-flash")  # 轻量任务（问题分类等）用小模型降延迟
EMBED_MODEL = os.getenv("FINQA_EMBED_MODEL", "text-embedding-v3")
RERANK_MODEL = os.getenv("FINQA_RERANK_MODEL", "gte-rerank-v2")
EMBED_DIM = 1024          # text-embedding-v3 默认维度
EMBED_BATCH = 10          # DashScope embedding 单次批量上限

# ---------- 解析与分块 ----------
# 解析后端: auto(优先 MinerU) | mineru | pymupdf
PARSE_BACKEND = os.getenv("FINQA_PARSE_BACKEND", "auto")
# 超长年报按 200 页分段提交给解析器，避免一次性解析失败
PARSE_PAGE_BATCH = 200
# 表格 LLM 序列化并发数（I/O 密集，并发可数倍提速入库）
TABLE_SERIAL_WORKERS = int(os.getenv("FINQA_TABLE_SERIAL_WORKERS", "6"))
# 云端 MinerU（mineru.net）精准解析 API：https://mineru.net 注册后申请 Token
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")
# 云端解析模型版本: vlm（官方推荐，质量高）| pipeline（更快）
MINERU_MODEL_VERSION = os.getenv("FINQA_MINERU_MODEL", "vlm")

CHUNK_SIZE = int(os.getenv("FINQA_CHUNK_SIZE", "500"))    # 小切片字符数
CHUNK_OVERLAP = int(os.getenv("FINQA_CHUNK_OVERLAP", "60"))
AB_CHUNK_SIZE = 800        # A/B 实验的对照切片大小
AB_CHUNK_OVERLAP = 100

# ---------- 检索 ----------
RECALL_TOP_K = 50          # 粗召回条数（30-50）
FINAL_TOP_K = 5             # Rerank 后保留条数
PARENT_PAGE_CAP = 8        # Small-to-Big 最多回捞的父页面数
# 混合检索权重: 向量 / BM25 / Doc2Query 问题扩展
W_VECTOR, W_BM25, W_QUERY = 0.45, 0.35, 0.20

# ---------- GraphRAG ----------
GRAPH_MAX_CHUNKS = 150      # 建图时最多使用的切片数（控制成本）
GRAPH_WORKERS = 4           # 实体抽取并发数
GLOBAL_COMMUNITY_TOP = 4    # 全局检索使用的社区摘要数

# ---------- 健康度 ----------
REQUIRED_INDICATORS = [
    "营业收入", "净利润", "毛利率", "资产负债率",
    "经营活动产生的现金流量净额", "总资产", "基本每股收益", "净资产收益率",
]
HEALTH_MISSING_THRESHOLD = 0.15   # 混合检索得分低于该值视为缺失

for _d in (DATA_DIR, COMPANIES_DIR, GRAPH_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
