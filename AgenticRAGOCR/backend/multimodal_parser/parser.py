import os
import json
import base64
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultimodalParser")

class MultimodalParser:
    """
    独立的多模态文档解析工具
    使用 DashScope (Qwen-VL) API 进行高精度解析
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set. Please provide it or set it as an environment variable.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = "qwen3-vl-flash-2026-01-22"

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _crop_image(self, image_path: str, bbox: List[int], output_path: str) -> bool:
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

    def parse_file(self, file_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        解析文档文件（图片或PDF）
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if output_dir is None:
            output_dir = file_path.parent / f"{file_path.stem}_parsed"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ext = file_path.suffix.lower()
        if ext in ['.pdf']:
            return self._parse_pdf(file_path, output_dir)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return [self._parse_image(file_path, 0, output_dir)]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _parse_pdf(self, pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """将 PDF 转换为图片并逐页解析"""
        results = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page = doc[i]
                image_path = output_dir / f"page_{i}.jpg"
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                pix.save(str(image_path))
                
                page_res = self._parse_image(image_path, i, output_dir)
                results.append(page_res)
                logger.info(f"Parsed page {i+1}/{len(doc)}")
            doc.close()
        except ImportError:
            logger.error("PyMuPDF (fitz) is required for PDF parsing. Install it with 'pip install pymupdf'.")
            raise
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
        
        return results

    def _parse_image(self, image_path: Path, page_index: int, output_dir: Path) -> Dict[str, Any]:
        """调用 Qwen-VL API 解析单张图片"""
        base64_image = self._encode_image(str(image_path))
        
        prompt = """你是一个专业的文档解析助手。请对这张文档图片进行深度解析。
你的任务是识别出图片中所有的内容块，包括：文本、表格、图像、公式、标题等。

要求：
1. 识别结果必须以 JSON 格式返回。
2. JSON 结构如下：
{
  "page_index": 数字,
  "parsing_res_list": [
    {
      "block_id": 数字,
      "block_label": "text" | "table" | "image" | "formula" | "paragraph_title" | "doc_title",
      "block_content": "内容（文本块直接填文本，表格填 Markdown 格式，公式填 LaTeX，图片/图表描述内容）",
      "block_bbox": [x1, y1, x2, y2], (归一化到 0-1000),
      "block_order": 数字
    },
    ...
  ]
}
3. 必须包含所有可见内容，禁止总结，必须逐行识别。
4. 所有的 block_bbox 必须准确覆盖该内容块。
5. 表格必须转换为 Markdown 格式。
6. 禁止输出任何解释性文字，只输出纯 JSON。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ],
                    }
                ],
                max_tokens=8192
            )

            content = response.choices[0].message.content
            # 清理 JSON 格式（去除 ```json ... ```）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            data["page_index"] = page_index
            
            # 处理图像块，提取子图
            for block in data.get("parsing_res_list", []):
                if block.get("block_label") == "image":
                    bbox = block.get("block_bbox")
                    if bbox and len(bbox) == 4:
                        img_name = f"img_p{page_index}_{block['block_id']}.jpg"
                        crop_path = output_dir / img_name
                        if self._crop_image(str(image_path), bbox, str(crop_path)):
                            block["image_save_path"] = str(crop_path)

            return data
        except Exception as e:
            logger.error(f"Error calling Qwen-VL API for page {page_index}: {e}")
            return {"page_index": page_index, "parsing_res_list": [], "error": str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parser.py <file_path> [api_key]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        parser = MultimodalParser(api_key=api_key)
        results = parser.parse_file(file_path)
        
        # 保存总结果
        output_file = Path(file_path).parent / f"{Path(file_path).stem}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully parsed {file_path}. Results saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
