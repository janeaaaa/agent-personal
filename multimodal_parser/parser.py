import os
import time
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from openai import OpenAI

try:
    from .schemas import ParsedBlock, OCRResult
    from .utils import encode_image, crop_image, normalize_text
except ImportError:
    from schemas import ParsedBlock, OCRResult
    from utils import encode_image, crop_image, normalize_text

logger = logging.getLogger(__name__)

class MultimodalDocParser:
    """
    高精度多模态文档解析工具
    集成 DashScope (Qwen-VL) API 
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-vl-max-latest"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model_name = model_name
        self.client = None
        
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            logger.warning("DASHSCOPE_API_KEY not set. API parsing will be unavailable.")

    async def parse(self, file_path: str, output_dir: Optional[str] = None) -> List[OCRResult]:
        """
        解析文档 (PDF 或图片)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = file_path.stem
        if output_dir is None:
            output_dir = file_path.parent / f"{doc_id}_parsed"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.client:
            return await self._parse_via_api(file_path, doc_id, output_dir)
        else:
            return await self._parse_fallback(file_path, doc_id, output_dir)

    async def _parse_via_api(self, file_path: Path, doc_id: str, output_dir: Path) -> List[OCRResult]:
        results = []
        
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(file_path))
                for i in range(len(doc)):
                    page = doc[i]
                    # 渲染为图片
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    temp_img_path = output_dir / f"page_{i}.png"
                    pix.save(str(temp_img_path))
                    
                    page_result = await self._parse_image_page(temp_img_path, doc_id, i, output_dir)
                    results.append(page_result)
                doc.close()
            except Exception as e:
                logger.error(f"Error parsing PDF via API: {e}")
                return await self._parse_fallback(file_path, doc_id, output_dir)
        else:
            # 单张图片
            page_result = await self._parse_image_page(file_path, doc_id, 0, output_dir)
            results.append(page_result)
            
        return results

    async def _parse_image_page(self, img_path: Path, doc_id: str, page_index: int, output_dir: Path) -> OCRResult:
        logger.info(f"Parsing page {page_index} via API...")
        
        prompt = """你是一个专业的文档高精度解析助手。请对这张图片进行极其详尽的 OCR 识别和布局分析。
要求：
1. **文字提取**：识别并原样提取图片中可见的所有文字。
2. **禁止总结**：必须原样提取文字，不得精简。
3. **逐行识别**：不要遗漏任何一行。
4. **布局块划分**：将文字划分为多个 "text" 块。
5. **表格解析**：将表格解析为 HTML 格式，标签为 "table"。
6. **图像识别**：描述视觉元素，标签为 "image"。
7. **坐标标注**：提供 [x1, y1, x2, y2] 坐标（0-1000 范围）。
8. **严格 JSON 格式**：只输出一个 JSON 列表。

输出示例：
[
  {"label": "text", "content": "内容...", "bbox": [100, 100, 900, 150], "order": 0},
  {"label": "table", "content": "<table>...</table>", "bbox": [100, 210, 900, 400], "order": 1}
]
"""
        base64_img = encode_image(str(img_path))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        start_time = time.time()
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        )
        
        blocks = []
        if completion and completion.choices:
            raw_content = completion.choices[0].message.content
            # 提取 JSON
            import re
            match = re.search(r'\[\s*\{.*\}\s*\]', raw_content, re.DOTALL)
            if match:
                json_content = match.group(0)
                try:
                    parsed_data = json.loads(json_content)
                    for idx, b in enumerate(parsed_data):
                        label = b.get("label", "text").lower()
                        content = b.get("content", "")
                        bbox = b.get("bbox", [0, 0, 0, 0])
                        order = b.get("order", idx)
                        
                        # 裁剪图片块 (如果是表格或图片)
                        img_rel_path = None
                        if label in ["table", "image"] and bbox != [0, 0, 0, 0]:
                            crop_name = f"crop_p{page_index}_b{idx}.png"
                            crop_path = output_dir / crop_name
                            if crop_image(str(img_path), bbox, str(crop_path)):
                                img_rel_path = str(crop_path)

                        blocks.append(ParsedBlock(
                            block_id=idx,
                            block_label=label,
                            block_content=normalize_text(content),
                            block_bbox=bbox,
                            block_order=order,
                            page_index=page_index,
                            image_path=img_rel_path
                        ))
                except Exception as e:
                    logger.error(f"Failed to parse JSON for page {page_index}: {e}")
        
        return OCRResult(
            doc_id=doc_id,
            page_index=page_index,
            blocks=blocks,
            total_blocks=len(blocks),
            processing_time=time.time() - start_time
        )

    async def _parse_fallback(self, file_path: Path, doc_id: str, output_dir: Path) -> List[OCRResult]:
        """简单的本地解析回退"""
        results = []
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(file_path))
                for i in range(len(doc)):
                    page = doc[i]
                    text = page.get_text()
                    blocks = [ParsedBlock(
                        block_id=0,
                        block_label="text",
                        block_content=normalize_text(text),
                        block_bbox=[0, 0, 1000, 1000],
                        block_order=0,
                        page_index=i
                    )]
                    results.append(OCRResult(
                        doc_id=doc_id,
                        page_index=i,
                        blocks=blocks,
                        total_blocks=1,
                        processing_time=0.0
                    ))
                doc.close()
            except:
                pass
        return results

    def save_results(self, results: List[OCRResult], output_path: str):
        """保存解析结果到 JSON"""
        data = [res.model_dump() for res in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 示例用法
    import asyncio
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python parser.py <file_path>")
            return

        parser = MultimodalDocParser()
        results = await parser.parse(sys.argv[1])
        
        output_file = Path(sys.argv[1]).stem + "_result.json"
        parser.save_results(results, output_file)
        print(f"Parsed {len(results)} pages. Result saved to {output_file}")

    asyncio.run(main())
