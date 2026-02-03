from pydantic import BaseModel
from typing import List, Optional

class ParsedBlock(BaseModel):
    """解析的文档块"""
    block_id: int
    block_label: str
    block_content: str
    block_bbox: List[int]  # [x1, y1, x2, y2]
    block_order: Optional[int] = None
    page_index: int = 0
    image_path: Optional[str] = None  # 块图片的路径

class OCRResult(BaseModel):
    """OCR 处理结果"""
    doc_id: str
    page_index: int
    blocks: List[ParsedBlock]
    total_blocks: int
    processing_time: float
