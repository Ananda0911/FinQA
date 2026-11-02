"""中文财报 PDF 解析：MinerU（本地/云端）/ PyMuPDF 多后端"""
from .base import ParsedDocument, Page, TableBlock
from .pymupdf_parser import PyMuPDFParser
from .table_serialize import serialize_tables

_backends = {}  # 按名称缓存，避免重复初始化


def _local_mineru():
    try:
        from .mineru_parser import MinerUParser
        p = MinerUParser()
        if p.available():
            return p
    except Exception:
        pass
    return None


def _cloud_mineru():
    from .. import config
    if config.MINERU_API_TOKEN:
        from .mineru_cloud import CloudMinerUParser
        return CloudMinerUParser()
    return None


def get_parser(backend: str = "auto"):
    """解析后端选择。

    - auto: 本地 MinerU → 云端 MinerU（已配 Token 时）→ PyMuPDF
    - mineru / mineru_cloud / pymupdf: 显式指定，不可用时直接报错（不静默降级）
    """
    if backend in _backends:
        return _backends[backend]

    if backend == "auto":
        p = _local_mineru() or _cloud_mineru() or PyMuPDFParser()
    elif backend == "mineru":
        p = _local_mineru()
        if p is None:
            raise RuntimeError("本地 MinerU 不可用：pip install \"mineru[core]\"，或改用 mineru_cloud / pymupdf")
    elif backend == "mineru_cloud":
        p = _cloud_mineru()
        if p is None:
            raise RuntimeError("未配置 MINERU_API_TOKEN（mineru.net 申请），无法使用云端 MinerU")
    elif backend == "pymupdf":
        p = PyMuPDFParser()
    else:
        raise ValueError(f"未知解析后端: {backend}")

    _backends[backend] = p
    return p
