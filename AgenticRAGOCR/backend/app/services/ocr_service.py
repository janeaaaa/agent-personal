"""
PaddleOCR Service for Document Parsing
"""

import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import re
import unicodedata
import base64
from typing import Tuple
from openai import OpenAI

from app.config import settings
from app.models.schemas import ParsedBlock, OCRResult, BlockType, DocumentStats

logger = logging.getLogger(__name__)

class OCRService:
    """PaddleOCR 文档解析服务"""

    def __init__(self):
        self.pipeline = None
        self.client = None
        self._initialized = False

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
        if settings.ocr_backend == "dashscope":
            logger.info("Using DashScope (Qwen-VL) for OCR (API mode)")
            self.pipeline = "dashscope" # 标记为 API 模式
            
            # 初始化 OpenAI 兼容客户端
            api_key = os.getenv("DASHSCOPE_API_KEY") or settings.dashscope_api_key
            if api_key:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            return

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
                        result = self.ocr(input, cls=True)
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

        # 准备输出目录
        if output_dir is None:
            output_dir = Path(settings.upload_dir) / doc_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 如果是 API 模式，直接调用 API 解析
        if self.pipeline == "dashscope":
            results = await self._dashscope_parse(file_path, doc_id, str(output_dir))
            self._save_results_to_files(results, output_dir)
            return results

        # 检查 pipeline 是否可用
        if self.pipeline is None:
            results = await self._fallback_parse(file_path, doc_id, str(output_dir))
            self._save_results_to_files(results, output_dir)
            return results

        try:
            start_time = time.time()

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

            # 统一保存解析结果到文件，方便用户验证
            self._save_results_to_files(results, output_dir)

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
                # 优先尝试使用 PyMuPDF (fitz) 提取带结构的文本块
                import fitz
                doc = fitz.open(file_path)
                for i in range(len(doc)):
                    page = doc[i]
                    # 获取文本块，包含基本的结构信息
                    text_blocks = page.get_text("blocks")
                    blocks = []
                    for idx, b in enumerate(text_blocks):
                        # b = (x0, y0, x1, y1, "text", block_no, block_type)
                        content = self._normalize_text(b[4])
                        if not content:
                            continue
                        blocks.append(ParsedBlock(
                            block_id=idx,
                            block_label="text",
                            block_content=content,
                            block_bbox=[int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                            block_order=idx,
                            page_index=i
                        ))
                    
                    if not blocks:
                        blocks = [ParsedBlock(
                            block_id=0,
                            block_label="text",
                            block_content="空白页面",
                            block_bbox=[0, 0, 0, 0],
                            block_order=0,
                            page_index=i
                        )]
                        
                    results.append(OCRResult(
                        doc_id=doc_id,
                        page_index=i,
                        blocks=blocks,
                        total_blocks=len(blocks),
                        processing_time=0.0
                    ))
                doc.close()
                return results
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed, falling back to PyPDF2: {e}")
                # 原有的 PyPDF2 逻辑作为二次回退
                try:
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
        """规范化文本内容"""
        if not s:
            return ""
        # 去除 Markdown 代码块标记
        s = re.sub(r'```(?:html|markdown|latex)?', '', s)
        s = s.replace('```', '')
        # 1. 规范化 Unicode (NFC)
        s = unicodedata.normalize("NFC", s)
        
        # 2. 移除不可见字符、控制字符以及私有区字符 (PUA)
        # 这里的正则包含了常见的控制字符和零宽字符
        s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200b-\u200f\ufeff\u202a-\u202e\u2060-\u206f]", "", s)
        
        # 使用列表推导式进一步过滤掉所有私有区字符 (Category 'Co')
        # 这样可以处理 􀤌 等 Qwen-VL 可能会产生的特殊编码
        s = "".join(c for c in s if unicodedata.category(c) != 'Co')
        
        # 3. 规范化空白字符 (保留换行)
        s = re.sub(r"[ \t\u00A0]+", " ", s)
        
        # 4. 移除行首尾空格
        lines = [line.strip() for line in s.split("\n")]
        return "\n".join(lines).strip()

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
            # 使用配置的布局检测模型路径
            engine = PPStructure(
                recovery=True, 
                image_orientation=True,
                layout_model_dir=settings.layout_detection_model_dir,
                show_log=False
            )
        except Exception:
            # 如果配置路径失败，尝试默认初始化
            try:
                engine = PPStructure(recovery=True, image_orientation=True, show_log=False)
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

    async def _dashscope_parse(self, file_path: str, doc_id: str, output_dir: str = "") -> List[OCRResult]:
        """使用阿里云百炼 Qwen-VL API 解析文档"""
        logger.info(f"Parsing document {doc_id} via DashScope API...")
        
        # 优先使用 settings 中的配置（已经由 pydantic-settings 处理了 .env 和 环境变量的优先级）
        api_key = settings.dashscope_api_key
        if not api_key:
            # 如果 settings 中没有，再尝试从环境变量直接读取
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            logger.error("DASHSCOPE_API_KEY is not set. Please set it in .env or as an environment variable.")
            return await self._fallback_parse(file_path, doc_id, "")
            
        try:
            import dashscope
            from dashscope import MultiModalConversation
            dashscope.api_key = api_key

            p = Path(file_path)
            results: List[OCRResult] = []

            # 对于 PDF，API 模式目前通过多页渲染或直接文件上传处理
            if p.suffix.lower() == ".pdf":
                try:
                    import fitz
                    from PIL import Image
                    import io
                    
                    doc = fitz.open(file_path)
                    # 处理所有页面
                    max_pages = len(doc)
                    for i in range(max_pages):
                        page = doc[i]
                        # 将 PDF 页面渲染为高分辨率图片
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("png")
                        
                        # 保存临时图片用于 API 调用
                        temp_img_path = Path(output_dir) / f"temp_page_{i}.png"
                        with open(temp_img_path, "wb") as f:
                            f.write(img_data)
                        
                        logger.info(f"Parsing PDF page {i} via {settings.qwen_vl_model_name}...")
                        prompt = f"""你是一个专业的文档高精度解析助手。请对这张图片进行极其详尽的 OCR 识别和布局分析。
        要求：
        1. **文字提取**：识别并原样提取图片中可见的所有文字。必须包含标题、正文、列表、页码、备注等所有文本。
        2. **禁止总结**：必须原样提取文字，不得进行任何形式的总结、精简或改写。
        3. **逐行识别**：不要遗漏任何一行文字。如果页面内容很多，请耐心地全部提取。
        4. **布局块划分**：将文字划分为多个 "text" 块，每个块对应一个段落或一个逻辑单元。
        5. **表格解析**：将所有表格解析为 HTML 格式。标签必须是 "table"。
        6. **图像识别**：识别图片中的视觉元素（图表、插图等），描述其内容。标签必须是 "image"。
        7. **坐标标注**：每个块都必须提供准确的 [x1, y1, x2, y2] 坐标（0-1000 范围）。
        8. **严格 JSON 格式**：只输出一个 JSON 列表，包含所有块。不要包含任何 markdown 标签或解释。必须严格遵守以下字段名：
           - "label": 块类型 ("text", "table", "image", "formula")
           - "content": 提取的内容
           - "bbox": [x1, y1, x2, y2] 坐标
           - "order": 阅读顺序索引
        
        输出示例：
        [
          {{"label": "text", "content": "第一段文字...", "bbox": [100, 100, 900, 150], "order": 0}},
          {{"label": "table", "content": "<table>...</table>", "bbox": [100, 210, 900, 400], "order": 1}}
        ]
        """
                        
                        base64_image = self._encode_image(str(temp_img_path))
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"
                                        }
                                    },
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ]
                        
                        loop = asyncio.get_event_loop()
                        completion = await loop.run_in_executor(
                            None,
                            lambda m=messages: self.client.chat.completions.create(
                                model=settings.qwen_vl_model_name,
                                messages=m,
                                max_tokens=settings.max_tokens,
                                extra_body={
                                     'enable_thinking': True,
                                     "thinking_budget": 8192
                                 }
                            )
                        )
                        
                        if completion and completion.choices:
                            msg = completion.choices[0].message
                            raw_content = ""
                            
                            # 提取思考过程 (可选)
                            reasoning = ""
                            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                                reasoning = msg.reasoning_content
                            
                            if reasoning:
                                logger.info(f"Page {i} reasoning length: {len(reasoning)}")
                            
                            # 提取最终内容
                            if hasattr(msg, 'content') and msg.content:
                                raw_content = msg.content
                            
                            # 打印原始响应内容长度
                            # logger.info(f"Page {i} RAW content length: {len(raw_content)}")
                            # if len(raw_content) < 500:
                            #     logger.info(f"Page {i} RAW content (short): {raw_content}")
                            # else:
                            #     logger.info(f"Page {i} RAW content (start): {raw_content[:200]}...")
                            #     logger.info(f"Page {i} RAW content (end): {raw_content[-200:]}")
                            print(f"\n--- DEBUG: Page {i} RAW content ---\n{raw_content}\n----------------------------\n")
                            
                            if not raw_content:
                                logger.error(f"Empty content from API for page {i}")
                                results.extend(await self._fallback_parse_single_page(file_path, doc_id, i))
                            else:
                                logger.info(f"Page {i} raw content length: {len(raw_content)}")
                                
                                # 清理 JSON 内容
                                json_content = raw_content.strip()
                                # 尝试正则表达式提取最外层的 []
                                import re
                                match = re.search(r'\[\s*\{.*\}\s*\]', json_content, re.DOTALL)
                                if match:
                                    json_content = match.group(0)
                                else:
                                    if json_content.startswith("```json"):
                                        json_content = json_content[7:]
                                    elif json_content.startswith("```"):
                                        json_content = json_content[3:]
                                    if json_content.endswith("```"):
                                        json_content = json_content[:-3]
                                json_content = json_content.strip()

                                try:
                                    parsed_blocks = json.loads(json_content)
                                    blocks = []
                                    block_id = 0
                                    for b in parsed_blocks:
                                        # 增加容错：处理不同的字段名 (label, content, bbox)
                                        # 有时模型会返回 bbox_2d 而不是 bbox
                                        # 有时模型会将内容放在 label 中（如果它误解了指令）
                                        
                                        b_label_raw = b.get("label", "text")
                                        b_content = b.get("content", "")
                                        b_bbox = b.get("bbox", b.get("bbox_2d", [0, 0, 0, 0]))
                                        b_order = b.get("order", block_id)

                                        # 特殊处理：如果 label 包含很多文字且 content 为空，可能是模型搞混了
                                        if len(b_label_raw) > 20 and not b_content:
                                            b_content = b_label_raw
                                            b_label = "text"
                                        else:
                                            b_label = b_label_raw.lower()

                                        # 映射中文标签或非标准标签
                                        if any(x in b_label for x in ["文本", "text", "文字"]):
                                            b_label = "text"
                                        elif any(x in b_label for x in ["表格", "table"]):
                                            b_label = "table"
                                        elif any(x in b_label for x in ["图片", "图像", "image", "figure"]):
                                            b_label = "image"
                                        elif any(x in b_label for x in ["公式", "formula", "equation"]):
                                            b_label = "formula"
                                        
                                        # 如果内容中还带有 "文字提取：" 等前缀，去掉它
                                        if isinstance(b_content, str):
                                            b_content = re.sub(r'^(文字提取|内容|提取|识别结果)[:：]\s*', '', b_content)
                                        
                                        # 如果是表格或图片，保存裁剪后的图片
                                        img_rel_path = None
                                        if b_label in ["table", "image"] and b_bbox != [0, 0, 0, 0]:
                                            crop_name = f"{doc_id}_p{i}_b{block_id}.png"
                                            crop_path = Path(output_dir) / crop_name
                                            if self._crop_image(str(temp_img_path), b_bbox, str(crop_path)):
                                                # 修正图片相对路径，确保符合 API 访问路径
                                                img_rel_path = f"/api/documents/{doc_id}/files/{crop_name}"
                                        
                                        blocks.append(ParsedBlock(
                                            block_id=block_id,
                                            block_label=b_label,
                                            block_content=b_content,
                                            block_bbox=b_bbox,
                                            block_order=b_order,
                                            page_index=i,
                                            image_path=img_rel_path
                                        ))
                                        block_id += 1
                                    
                                    results.append(OCRResult(
                                        doc_id=doc_id,
                                        page_index=i,
                                        blocks=blocks,
                                        total_blocks=len(blocks),
                                        processing_time=0.0
                                    ))
                                except Exception as je:
                                    logger.error(f"Failed to parse JSON for page {i}: {je}")
                                    # 尝试基础提取作为备选
                                    results.extend(await self._fallback_parse_single_page(file_path, doc_id, i))
                        else:
                            logger.error(f"DashScope API error on page {i}: {completion}")
                            # 单页失败时尝试基础提取
                            results.extend(await self._fallback_parse_single_page(file_path, doc_id, i))
                        
                        # 删除临时图片
                        if temp_img_path.exists():
                            temp_img_path.unlink()
                    
                    doc.close()
                    return results
                except Exception as e:
                    logger.warning(f"Qwen-VL PDF rendering failed, falling back to text extraction: {e}")
                    return await self._fallback_parse(file_path, doc_id, "")

            # 如果是图片
            prompt = """你是一个专业的文档高精度解析助手。请对这张图片进行极其详尽的 OCR 识别和布局分析。
要求：
1. **完整性优先**：识别并提取图片中可见的所有文字内容，严禁遗漏任何段落、标题、页眉、页脚或注脚。
2. **禁止总结**：必须原样提取文字，不得进行任何形式的总结、精简或改写。
3. **布局分析**：保持原始阅读顺序，将文字划分为合理的语义块（text）。
4. **表格处理**：识别并解析所有表格，输出为标准的 HTML 表格格式，必须包含所有单元格内容。标签必须是 "table"。
5. **视觉元素**：识别图片中的图形、图像、图表、照片等，标记为 "image" 标签，并对其内容进行详细描述。
6. **数学公式**：识别数学公式，输出为 LaTeX 格式。标签必须是 "formula"。
7. **输出格式**：必须严格按以下 JSON 列表格式输出。不要输出任何其他解释文字、markdown 标签或思考过程。

输出示例：
[
  {"label": "text", "content": "完整文本内容...", "bbox": [x1, y1, x2, y2], "order": 0},
  {"label": "table", "content": "<table>...</table>", "bbox": [x1, y1, x2, y2], "order": 1},
  {"label": "image", "content": "对图像内容的详细描述...", "bbox": [x1, y1, x2, y2], "order": 2}
]
注意：
- 坐标范围为 [0, 1000, 0, 1000] (x1, y1, x2, y2)。
- 确保 JSON 格式严格正确，如果内容很多，请确保输出完整，不要中途停止。
"""

            base64_image = self._encode_image(str(file_path))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=settings.qwen_vl_model_name,
                    messages=messages,
                    max_tokens=settings.max_tokens,
                    extra_body={
                        'enable_thinking': True,
                        "thinking_budget": 8192
                    }
                )
            )

            if completion and completion.choices:
                msg = completion.choices[0].message
                raw_content = ""
                
                # 提取思考过程 (可选)
                reasoning = ""
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    reasoning = msg.reasoning_content
                
                if reasoning:
                    logger.info(f"Image reasoning length: {len(reasoning)}")
                
                # 提取最终内容
                if hasattr(msg, 'content') and msg.content:
                    raw_content = msg.content
                
                if not raw_content:
                    logger.error("Empty content from API for image")
                    return await self._fallback_parse(file_path, doc_id, "")

                logger.info(f"Image raw content length: {len(raw_content)}")
                
                # 清理 JSON 内容
                json_content = raw_content.strip()
                # 尝试正则表达式提取最外层的 []
                import re
                match = re.search(r'\[\s*\{.*\}\s*\]', json_content, re.DOTALL)
                if match:
                    json_content = match.group(0)
                else:
                    if json_content.startswith("```json"):
                        json_content = json_content[7:]
                    elif json_content.startswith("```"):
                        json_content = json_content[3:]
                    if json_content.endswith("```"):
                        json_content = json_content[:-3]
                json_content = json_content.strip()

                try:
                    parsed_blocks = json.loads(json_content)
                    blocks = []
                    block_id = 0
                    for b in parsed_blocks:
                        b_label = b.get("label", "text")
                        b_content = b.get("content", "")
                        b_bbox = b.get("bbox", [0, 0, 0, 0])
                        b_order = b.get("order", block_id)
                        
                        # 如果是表格或图片，保存裁剪后的图片
                        img_rel_path = None
                        if b_label in ["table", "image"] and b_bbox != [0, 0, 0, 0]:
                            crop_name = f"{doc_id}_b{block_id}.png"
                            crop_path = Path(output_dir) / crop_name
                            if self._crop_image(str(file_path), b_bbox, str(crop_path)):
                                img_rel_path = f"/api/documents/{doc_id}/files/{crop_name}"
                        
                        blocks.append(ParsedBlock(
                            block_id=block_id,
                            block_label=b_label,
                            block_content=b_content,
                            block_bbox=b_bbox,
                            block_order=b_order,
                            page_index=0,
                            image_path=img_rel_path
                        ))
                        block_id += 1
                    
                    results.append(OCRResult(
                        doc_id=doc_id,
                        page_index=0,
                        blocks=blocks,
                        total_blocks=len(blocks),
                        processing_time=0.0
                    ))
                except Exception as je:
                    logger.error(f"Failed to parse JSON for image: {je}")
                    return await self._fallback_parse(file_path, doc_id, "")
            else:
                logger.error(f"DashScope API error: {completion}")
                return await self._fallback_parse(file_path, doc_id, "")

            return results
        except Exception as e:
            logger.error(f"Failed to parse via DashScope: {e}")
            return await self._fallback_parse(file_path, doc_id, "")

    async def _fallback_parse_single_page(self, file_path: str, doc_id: str, page_index: int) -> List[OCRResult]:
        """回退解析单页（内部辅助）"""
        try:
            import fitz
            doc = fitz.open(file_path)
            if page_index >= len(doc):
                return []
            page = doc[page_index]
            text_blocks = page.get_text("blocks")
            blocks = []
            for idx, b in enumerate(text_blocks):
                content = self._normalize_text(b[4])
                if not content: continue
                blocks.append(ParsedBlock(
                    block_id=idx, block_label="text", block_content=content,
                    block_bbox=[int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                    block_order=idx, page_index=page_index
                ))
            doc.close()
            return [OCRResult(doc_id=doc_id, page_index=page_index, blocks=blocks, total_blocks=len(blocks), processing_time=0.0)]
        except Exception:
            return []

    def _save_results_to_files(self, results: List[OCRResult], output_dir: Path):
        """将解析结果保存到文件，方便验证"""
        try:
            # 1. 保存为全量 JSON
            json_data = [res.model_dump() for res in results]
            with open(output_dir / "full_res.json", "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            # 2. 保存为 Markdown (所有页面合并)
            md_content = []
            for res in results:
                md_content.append(f"## Page {res.page_index + 1}\n")
                for block in res.blocks:
                    if block.block_label == "table":
                        md_content.append(f"{block.block_content}\n")
                    else:
                        md_content.append(f"{block.block_content}\n\n")
            
            with open(output_dir / "full_res.md", "w", encoding="utf-8") as f:
                f.write("".join(md_content))
                
            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save results to files: {e}")

    async def cleanup(self):
        """清理资源"""
        self.pipeline = None
        self._initialized = False
        logger.info("OCR service cleaned up")
