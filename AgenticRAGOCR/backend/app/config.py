"""
Application Configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # 阿里云百炼配置
    dashscope_api_key: str = ""
    qwen_model_name: str = "qwen3-max-2026-01-23"
    qwen_vl_model_name: str = "qwen3-vl-flash-2026-01-22"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 8192

    # 向量数据库配置
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "text-embedding-v3"  # Qwen embedding 模型

    # RAG 配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # OCR 配置
    ocr_backend: str = "dashscope"  # 默认使用云端 API，提供更好的视觉识别能力
    
    # PaddleOCR 模型路径
    paddleocr_vl_model_dir: str = "./models/PaddleOCR-VL-0.9B"
    layout_detection_model_dir: str = "./models/PP-DocLayoutV2"

    # 状态存储（Redis 可选）
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # 文件上传配置
    upload_dir: str = "./uploads"
    max_upload_size: int = 50  # MB
    allowed_extensions: List[str] = ["pdf", "png", "jpg", "jpeg"]

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"]

    # 日志配置
    log_level: str = "INFO"

    @field_validator('allowed_extensions', 'cors_origins', mode='before')
    @classmethod
    def parse_list_from_str(cls, v):
        """将逗号分隔的字符串转换为列表"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保必要的目录存在
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        
        # 处理相对路径为绝对路径
        base_dir = Path(__file__).resolve().parent.parent
        if self.paddleocr_vl_model_dir.startswith("./"):
            self.paddleocr_vl_model_dir = str(base_dir / self.paddleocr_vl_model_dir[2:])
        if self.layout_detection_model_dir.startswith("./"):
            self.layout_detection_model_dir = str(base_dir / self.layout_detection_model_dir[2:])
        
        # 确保模型目录存在（如果不存在则创建，提醒用户放置模型）
        Path(self.paddleocr_vl_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.layout_detection_model_dir).mkdir(parents=True, exist_ok=True)


# 全局设置实例
settings = Settings()
