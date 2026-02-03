"""
PaddleOCR Service for Document Parsing
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import re
import unicodedata
from typing import Tuple

from app.config import settings
from app.models.schemas import ParsedBlock, OCRResult, BlockType, DocumentStats

logger = logging.getLogger(__name__)

class OCRService:
    """PaddleOCR 文档解析服务"""

    def __init__(self):
        self.pipeline = None
        self._initialized = False

    async def initialize(self):
        """初始化 PaddleOCR pipeline"""
        if self._initialized:
            return

        try:
            logger.info("Initializing PaddleOCR pipeline...")

            # 在线程池中初始化（避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_pipeline)

            if self.pipeline is not None:
                self._initialized = True
                logger.info("PaddleOCR pipeline initialized successfully")
            else:
                raise Exception("Pipeline initialization returned None")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            logger.warning("PaddleOCR initialization failed, service will not be available")
            self._initialized = False
            self.pipeline = None

    def _init_pipeline(self):
        """初始化 pipeline（同步）"""
        try:
            from paddleocr import PaddleOCRVL
            self.pipeline = PaddleOCRVL(
                vl_rec_model_dir=settings.paddleocr_vl_model_dir,
                layout_detection_model_dir=settings.layout_detection_model_dir
            )
            logger.info(f"PaddleOCRVL loaded with model: {settings.paddleocr_vl_model_dir}")
        except Exception as e:
            try:
                from paddleocr import PaddleOCR
                class BasicPaddleOCRPipeline:
                    def __init__(self):
                        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                    def predict(self, input: str, save_path: str):
                        p = Path(input)
                        if p.suffix.lower() == ".pdf":
                            pages = []
                            try:
                                import PyPDF2
                                with open(input, "rb") as f:
                                    reader = PyPDF2.PdfReader(f)
                                    for i in range(len(reader.pages)):
                                        page = reader.pages[i]
                                text = page.extract_text() or ""
                                content = text.strip() if text else ""
                                content = self._normalize_text(content)
                                        if not content:
                                            content = ""
                                        parsing_res_list = []
                                        if content:
                                            parsing_res_list.append({
                                                "block_id": 0,
                                                "block_label": "text",
                                                "block_content": content,
                                                "block_bbox": [0, 0, 0, 0],
                                                "block_order": 0
                                            })
                                        pages.append({
                                            "page_index": i,
                                            "parsing_res_list": parsing_res_list
                                        })
                                return pages
                            except Exception:
                                return [{
                                    "page_index": 0,
                                    "parsing_res_list": []
                                }]
                        result = self.ocr.ocr(input, cls=True)
                        blocks = []
                        block_id = 0
                        if result and len(result) > 0:
                            lines = result[0]
                            for idx, line in enumerate(lines):
                                try:
                                    box = line[0]
                                    text = line[1][0]
                                    xs = [pt[0] for pt in box]
                                    ys = [pt[1] for pt in box]
                                    x1, y1 = int(min(xs)), int(min(ys))
                                    x2, y2 = int(max(xs)), int(max(ys))
                                    blocks.append({
                                        "block_id": block_id,
                                        "block_label": "text",
                                        "block_content": text,
                                        "block_bbox": [x1, y1, x2, y2],
                                        "block_order": idx
                                    })
                                    block_id += 1
                                except Exception:
                                    continue
                        return [{
                            "page_index": 0,
                            "parsing_res_list": blocks
                        }]
                self.pipeline = BasicPaddleOCRPipeline()
                logger.info("PaddleOCRVL unavailable, using Basic PaddleOCR pipeline")
            except Exception as e2:
                logger.error(f"Error in _init_pipeline: {e2}")
                self.pipeline = None
                raise

    async def parse_document(
        self,
        file_path: str,
        doc_id: str,
        output_dir: Optional[str] = None
    ) -> List[OCRResult]:
        """
        解析文档并返回结构化结果

        Args:
            file_path: 文档路径
            doc_id: 文档ID
            output_dir: 输出目录（保存JSON、Markdown等）

        Returns:
            每页的 OCR 结果列表
        """
        if not self._initialized:
            await self.initialize()

        # 检查 pipeline 是否可用
        if self.pipeline is None:
            return await self._fallback_parse(file_path, doc_id, output_dir)

        try:
            start_time = time.time()

            # 准备输出目录
            if output_dir is None:
                output_dir = Path(settings.upload_dir) / doc_id
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 在线程池中执行 OCR（避免阻塞）
            loop = asyncio.get_event_loop()
            ocr_outputs = await loop.run_in_executor(
                None,
                self._run_ocr,
                file_path,
                str(output_dir)
            )

            # 从保存的JSON文件解析结果(而不是从内存对象)
            results = []
            json_files = sorted(output_dir.glob("*_res.json"))

            if json_files:
                # 从JSON文件读取
                import json
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    ocr_result = self._parse_json_output(json_data, doc_id)
                    results.append(ocr_result)
                    logger.info(f"Parsed {json_file.name}: {len(ocr_result.blocks)} blocks")
            else:
                # Fallback: 从内存对象解析
                for res in ocr_outputs:
                    ocr_result = self._parse_ocr_output(res, doc_id)
                    results.append(ocr_result)

            processing_time = time.time() - start_time
            logger.info(
                f"Document {doc_id} parsed: {len(results)} pages, "
                f"time: {processing_time:.2f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Error parsing document {doc_id}: {e}")
            raise

    async def _fallback_parse(self, file_path: str, doc_id: str, output_dir: str) -> List[OCRResult]:
        results: List[OCRResult] = []
        p = Path(file_path)
        ext = p.suffix.lower()
        page_index = 0
        blocks: List[ParsedBlock] = []
        if ext == ".pdf":
            try:
                # 先尝试使用 PyPDF2 提取文本
                import PyPDF2
                extracted_pages: List[str] = []
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for i in range(len(reader.pages)):
                        page = reader.pages[i]
                        text = page.extract_text() or ""
                        content = text.strip() if text else ""
                        content = self._normalize_text(content)
                        extracted_pages.append(content)
                # 如果 PyPDF2 没有提取到有效文本，尝试 pdfminer 回退
                if not any(extracted_pages):
                    try:
                        from pdfminer.high_level import extract_text
                        full_text = extract_text(file_path) or ""
                        # 简单按换行切页（pdfminer不保留页边界时）
                        if full_text:
                            # 以长度片段近似分页，避免全量落在一页
                            chunk_len = max(1000, len(full_text) // 5)
                            tmp_pages = []
                            for i in range(0, len(full_text), chunk_len):
                                c = full_text[i:i + chunk_len].strip()
                                tmp_pages.append(self._normalize_text(c))
                            extracted_pages = tmp_pages
                        else:
                            extracted_pages = []
                    except Exception:
                        extracted_pages = []
                if not extracted_pages:
                    extracted_pages = ["空白页面"]
                for i, content in enumerate(extracted_pages):
                    if not content:
                        content = "空白页面"
                    blocks = [
                        ParsedBlock(
                            block_id=0,
                            block_label="text",
                            block_content=content,
                            block_bbox=[0, 0, 0, 0],
                            block_order=0,
                            page_index=i
                        )
                    ]
                    results.append(OCRResult(
                        doc_id=doc_id,
                        page_index=i,
                        blocks=blocks,
                        total_blocks=len(blocks),
                        processing_time=0.0
                    ))
            except Exception:
                blocks = [
                    ParsedBlock(
                        block_id=0,
                        block_label="text",
                        block_content=f"文件 {p.name}",
                        block_bbox=[0, 0, 0, 0],
                        block_order=0,
                        page_index=page_index
                    )
                ]
                results.append(OCRResult(
                    doc_id=doc_id,
                    page_index=page_index,
                    blocks=blocks,
                    total_blocks=len(blocks),
                    processing_time=0.0
                ))
        elif ext in [".png", ".jpg", ".jpeg"]:
            try:
                try:
                    blocks = self._ppstructure_parse_image(str(p))
                except Exception:
                    blocks = self._easyocr_parse_image(str(p))
                results.append(OCRResult(
                    doc_id=doc_id,
                    page_index=page_index,
                    blocks=blocks,
                    total_blocks=len(blocks),
                    processing_time=0.0
                ))
            except Exception:
                blocks = [
                    ParsedBlock(
                        block_id=0,
                        block_label="image",
                        block_content=f"图像 {p.name}",
                        block_bbox=[0, 0, 0, 0],
                        block_order=0,
                        page_index=page_index
                    )
                ]
                results.append(OCRResult(
                    doc_id=doc_id,
                    page_index=page_index,
                    blocks=blocks,
                    total_blocks=len(blocks),
                    processing_time=0.0
                ))
        else:
            blocks = [
                ParsedBlock(
                    block_id=0,
                    block_label="image",
                    block_content=f"图像 {p.name}",
                    block_bbox=[0, 0, 0, 0],
                    block_order=0,
                    page_index=page_index
                )
            ]
            results.append(OCRResult(
                doc_id=doc_id,
                page_index=page_index,
                blocks=blocks,
                total_blocks=len(blocks),
                processing_time=0.0
            ))
        return results

    def _run_ocr(self, file_path: str, output_dir: str):
        """运行 OCR（同步）"""
        output = self.pipeline.predict(
            input=file_path,
            save_path=output_dir
        )

        # 保存结果到文件
        for res in output:
            if hasattr(res, "save_to_json"):
                res.save_to_json(save_path=output_dir)
            if hasattr(res, "save_to_markdown"):
                res.save_to_markdown(save_path=output_dir)
            if hasattr(res, "save_to_img"):
                res.save_to_img(save_path=output_dir)

        return output

    def _parse_json_output(self, json_data: dict, doc_id: str) -> OCRResult:
        """从JSON数据解析OCR结果"""
        # 确保page_index是整数
        page_index = json_data.get('page_index')
        if page_index is None:
            page_index = 0

        parsing_results = json_data.get('parsing_res_list', [])

        # 转换为 ParsedBlock
        blocks = []
        for item in parsing_results:
            try:
                # 确保block_order是整数或None
                block_order = item.get('block_order')
                if block_order is not None and not isinstance(block_order, int):
                    try:
                        block_order = int(block_order)
                    except:
                        block_order = None

                block = ParsedBlock(
                    block_id=item.get('block_id', 0),
                    block_label=item.get('block_label', 'text'),
                    block_content=item.get('block_content', ''),
                    block_bbox=item.get('block_bbox', [0, 0, 0, 0]),
                    block_order=block_order,
                    page_index=page_index
                )
                blocks.append(block)
            except Exception as e:
                logger.warning(f"Failed to parse block: {e}")
                continue

        return OCRResult(
            doc_id=doc_id,
            page_index=page_index,
            blocks=blocks,
            total_blocks=len(blocks),
            processing_time=0.0
        )

    def _parse_ocr_output(self, res, doc_id: str) -> OCRResult:
        """解析单页 OCR 输出(从内存对象)"""
        page_data = res if isinstance(res, dict) else (res.__dict__ if hasattr(res, '__dict__') else {})

        page_index = page_data.get('page_index', 0)
        parsing_results = page_data.get('parsing_res_list', [])

        # 转换为 ParsedBlock
        blocks = []
        for item in parsing_results:
            try:
                block = ParsedBlock(
                    block_id=item.get('block_id', 0),
                    block_label=item.get('block_label', 'text'),
                    block_content=item.get('block_content', ''),
                    block_bbox=item.get('block_bbox', [0, 0, 0, 0]),
                    block_order=item.get('block_order'),
                    page_index=page_index
                )
                blocks.append(block)
            except Exception as e:
                logger.warning(f"Failed to parse block: {e}")
                continue

        return OCRResult(
            doc_id=doc_id,
            page_index=page_index,
            blocks=blocks,
            total_blocks=len(blocks),
            processing_time=0.0
        )

    def calculate_stats(self, ocr_results: List[OCRResult]) -> DocumentStats:
        """计算文档统计信息"""
        stats = DocumentStats(
            doc_id=ocr_results[0].doc_id if ocr_results else "",
            text_blocks=0,
            table_blocks=0,
            image_blocks=0,
            formula_blocks=0,
            total_blocks=0
        )

        for result in ocr_results:
            for block in result.blocks:
                stats.total_blocks += 1

                label = block.block_label.lower()
                if 'table' in label:
                    stats.table_blocks += 1
                elif any(x in label for x in ['image', 'figure', 'chart']):
                    stats.image_blocks += 1
                elif 'formula' in label or 'equation' in label:
                    stats.formula_blocks += 1
                else:
                    stats.text_blocks += 1

        return stats

    def get_block_type(self, label: str) -> str:
        """获取块类型的统一标签"""
        label = label.lower()

        if 'table' in label:
            return 'table'
        elif any(x in label for x in ['image', 'figure', 'chart']):
            return 'image'
        elif 'formula' in label or 'equation' in label:
            return 'formula'
        else:
            return 'text'

    def _normalize_text(self, s: str) -> str:
        s = unicodedata.normalize("NFC", s or "")
        s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", s)
        s = re.sub(r"[ \t\u00A0]+", " ", s)
        return s.strip()

    def _easyocr_parse_image(self, file_path: str) -> List[ParsedBlock]:
        try:
            import easyocr
        except Exception:
            raise
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        result = reader.readtext(file_path, detail=1)
        blocks: List[ParsedBlock] = []
        block_id = 0
        for idx, item in enumerate(result):
            try:
                box, text, _ = item
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                content = self._normalize_text(text or "")
                if not content:
                    continue
                blocks.append(ParsedBlock(
                    block_id=block_id,
                    block_label="text",
                    block_content=content,
                    block_bbox=[x1, y1, x2, y2],
                    block_order=idx,
                    page_index=0
                ))
                block_id += 1
            except Exception:
                continue
        if not blocks:
            blocks = [ParsedBlock(
                block_id=0,
                block_label="image",
                block_content=f"图像 {Path(file_path).name}",
                block_bbox=[0, 0, 0, 0],
                block_order=0,
                page_index=0
            )]
        return blocks

    def _ppstructure_parse_image(self, file_path: str) -> List[ParsedBlock]:
        try:
            from paddleocr import PPStructure
        except Exception:
            raise
        try:
            engine = PPStructure(recovery=True, image_orientation=True)
        except Exception:
            raise
        try:
            result = engine(file_path)
        except Exception:
            raise
        blocks: List[ParsedBlock] = []
        block_id = 0
        for idx, item in enumerate(result):
            try:
                label = item.get('type', 'text')
                content = self._normalize_text(item.get('res', '')) if isinstance(item.get('res'), str) else ""
                bbox = item.get('bbox', [0, 0, 0, 0])
                if label == 'table':
                    blocks.append(ParsedBlock(
                        block_id=block_id,
                        block_label="table",
                        block_content=content or "[表格]",
                        block_bbox=bbox if isinstance(bbox, list) and len(bbox) == 4 else [0, 0, 0, 0],
                        block_order=idx,
                        page_index=0
                    ))
                elif label in ['figure', 'image']:
                    blocks.append(ParsedBlock(
                        block_id=block_id,
                        block_label="image",
                        block_content=content or "[图像]",
                        block_bbox=bbox if isinstance(bbox, list) and len(bbox) == 4 else [0, 0, 0, 0],
                        block_order=idx,
                        page_index=0
                    ))
                else:
                    if not content:
                        continue
                    blocks.append(ParsedBlock(
                        block_id=block_id,
                        block_label="text",
                        block_content=content,
                        block_bbox=bbox if isinstance(bbox, list) and len(bbox) == 4 else [0, 0, 0, 0],
                        block_order=idx,
                        page_index=0
                    ))
                block_id += 1
            except Exception:
                continue
        if not blocks:
            blocks = [ParsedBlock(
                block_id=0,
                block_label="image",
                block_content=f"图像 {Path(file_path).name}",
                block_bbox=[0, 0, 0, 0],
                block_order=0,
                page_index=0
            )]
        return blocks

    async def cleanup(self):
        """清理资源"""
        self.pipeline = None
        self._initialized = False
        logger.info("OCR service cleaned up")
