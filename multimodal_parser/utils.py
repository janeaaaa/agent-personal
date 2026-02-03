import base64
import logging
import re
import unicodedata
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    """将图片编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def crop_image(image_path: str, bbox: List[int], output_path: str) -> bool:
    """根据 bbox 裁剪图片"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            # bbox 格式 [x1, y1, x2, y2], 范围 [0, 1000]
            left = bbox[0] * width / 1000
            top = bbox[1] * height / 1000
            right = bbox[2] * width / 1000
            bottom = bbox[3] * height / 1000
            
            # 确保坐标合法
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(left + 1, min(right, width))
            bottom = max(top + 1, min(bottom, height))
            
            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_path)
            return True
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return False

def normalize_text(s: str) -> str:
    """规范化文本内容"""
    if not s:
        return ""
    # 去除 Markdown 代码块标记
    s = re.sub(r'```(?:html|markdown|latex)?', '', s)
    s = s.replace('```', '')
    # 1. 规范化 Unicode (NFC)
    s = unicodedata.normalize("NFC", s)
    
    # 2. 移除不可见字符、控制字符以及私有区字符 (PUA)
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200b-\u200f\ufeff\u202a-\u202e\u2060-\u206f]", "", s)
    
    # 使用列表推导式进一步过滤掉所有私有区字符 (Category 'Co')
    s = "".join(c for c in s if unicodedata.category(c) != 'Co')
    
    # 3. 规范化空白字符 (保留换行)
    s = re.sub(r"[ \t\u00A0]+", " ", s)
    
    # 4. 移除行首尾空格
    lines = [line.strip() for line in s.split("\n")]
    return "\n".join(lines).strip()
