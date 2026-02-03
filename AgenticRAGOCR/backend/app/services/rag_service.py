"""
RAG Service for Document Indexing and Retrieval
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import os

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
from dashscope import TextEmbedding
import hashlib
import re
import unicodedata

from app.config import settings
from app.models.schemas import OCRResult, ParsedBlock, Citation, DocumentStats

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务 - 文档索引与检索 (恢复原始版本逻辑)"""

    def __init__(self):
        self.chroma_client = None
        self.text_splitter = None
        self._initialized = False
        self.collections = {}
        self.collection_dims = {}
        self.embedding_batch_size = 10  # Qwen API 批次大小限制（最大10）

    async def initialize(self):
        """初始化 RAG 服务"""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG service...")

            # 在线程池中初始化（避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_components)

            self._initialized = True
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise

    def _init_components(self):
        """初始化组件（同步）"""
        # 初始化 ChromaDB
        chroma_dir = Path(settings.chroma_persist_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # 彻底禁用 ChromaDB 遥测
        os.environ["CHROMA_TELEMETRY"] = "False"
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 设置 DashScope API Key（用于 Embedding）
        dashscope.api_key = settings.dashscope_api_key
        logger.info(f"Using Qwen embedding model: {settings.embedding_model}")

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    async def index_document(
        self,
        doc_id: str,
        ocr_results: List[OCRResult],
        file_name: str
    ) -> DocumentStats:
        """
        索引文档到向量数据库
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(f"Indexing document {doc_id}...")

            # 创建文档专属的 collection
            collection_name = f"doc_{doc_id}"

            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None,
                self._get_or_create_collection,
                collection_name
            )

            self.collections[doc_id] = collection

            # 准备索引数据
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []

            chunk_id = 0
            stats = DocumentStats(
                doc_id=doc_id,
                text_blocks=0,
                table_blocks=0,
                image_blocks=0,
                formula_blocks=0,
                total_blocks=0
            )

            for ocr_result in ocr_results:
                page_index = ocr_result.page_index
                
                # --- 优化 D: 块合并逻辑 (Block Merging) ---
                merged_blocks = self._merge_blocks_spatial(ocr_result.blocks)
                
                for block in merged_blocks:
                    stats.total_blocks += 1

                    # 统计块类型
                    block_type = self._get_block_type(block.block_label)
                    if block_type == "text":
                        stats.text_blocks += 1
                    elif block_type == "table":
                        stats.table_blocks += 1
                    elif block_type == "image":
                        stats.image_blocks += 1
                    elif block_type == "formula":
                        stats.formula_blocks += 1

                    # --- 优化 C: 向量化规则优化 - 上下文路径注入 ---
                    # 生成上下文前缀，增强语义指向性
                    context_prefix = f"[文档:{file_name} | 第{page_index+1}页 | 类型:{block_type}] "
                    
                    # 准备块的元数据
                    base_metadata = {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_index": page_index,
                        "block_id": block.block_id,
                        "block_type": block_type,
                        "block_label": block.block_label,
                        "block_bbox": json.dumps(block.block_bbox),
                        "block_order": block.block_order if block.block_order is not None else 0,
                        "full_block_content": block.block_content # 保存完整内容用于召回
                    }

                    content = block.block_content.strip()
                    if not content:
                        continue
                    content = self._preprocess_for_index(content)

                    # --- 优化 B: 分块逻辑优化 - 父子块逻辑 ---
                    # 1. 设定阈值为 400 (用户要求)
                    # 2. 设定重叠度为 10% (40 字符，恢复以提高准确性)
                    sub_chunk_size = 400
                    sub_chunk_overlap = 40
                    
                    if block_type == "text" and len(content) > sub_chunk_size:
                        # 使用较小的细粒度分块
                        child_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=sub_chunk_size,
                            chunk_overlap=sub_chunk_overlap,
                            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
                        )
                        chunks = child_splitter.split_text(content)

                        for i, chunk in enumerate(chunks):
                            metadata = base_metadata.copy()
                            metadata["chunk_index"] = i
                            metadata["total_chunks"] = len(chunks)
                            metadata["is_child"] = True
                            
                            # 将上下文前缀注入文本，增强向量语义
                            enhanced_content = context_prefix + chunk
                            
                            all_chunks.append(enhanced_content)
                            all_metadatas.append(metadata)
                            all_ids.append(f"{doc_id}_p{page_index}_b{block.block_id}_c{i}")
                            chunk_id += 1
                    else:
                        # 表格、图片、短文本等不进一步分块，作为独立单元
                        metadata = base_metadata.copy()
                        metadata["is_child"] = False
                        
                        # 同样注入上下文前缀
                        enhanced_content = context_prefix + content
                        
                        all_chunks.append(enhanced_content)
                        all_metadatas.append(metadata)
                        all_ids.append(f"{doc_id}_p{page_index}_b{block.block_id}")
                        chunk_id += 1

            # 批量生成 embeddings
            if all_chunks:
                logger.info(f"Generating embeddings for {len(all_chunks)} chunks using Qwen...")
                all_embeddings = await self._generate_embeddings(all_chunks, target_dim=1024)
                try:
                    dim = len(all_embeddings[0]) if all_embeddings else 0
                    if dim:
                        self.collection_dims[doc_id] = dim
                except Exception:
                    pass

                # 批量插入到 ChromaDB
                logger.info(f"Inserting {len(all_chunks)} chunks into ChromaDB...")
                await loop.run_in_executor(
                    None,
                    collection.add,
                    all_ids,
                    all_embeddings,
                    all_metadatas,
                    all_chunks
                )

            logger.info(f"Document {doc_id} indexed successfully: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            raise

    def _preprocess_for_index(self, s: str) -> str:
        """索引前的文本预处理 (恢复原始简洁版本)"""
        if not s:
            return ""
        s = unicodedata.normalize("NFC", s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t\u00A0]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"-\n", "", s)
        return s.strip()

    def _merge_blocks_spatial(self, blocks: List[ParsedBlock]) -> List[ParsedBlock]:
        """
        基于“空间邻近性 + 结构启发式”的块合并增强逻辑
        """
        if not blocks:
            return []

        # 按 block_order 或 y 坐标排序
        sorted_blocks = sorted(blocks, key=lambda b: (b.block_order if b.block_order is not None else 0, b.block_bbox[1]))
        
        merged = []
        if not sorted_blocks:
            return merged
            
        import copy
        import re
        current_block = copy.deepcopy(sorted_blocks[0])
        
        # 标点符号正则表达式：用于判断句子是否结束
        sentence_ends = re.compile(r'[。！？.!?：:；;]$')
        
        for i in range(1, len(sorted_blocks)):
            next_block = copy.deepcopy(sorted_blocks[i])
            
            # 只合并 text 类型
            if current_block.block_label == "text" and next_block.block_label == "text":
                curr_bbox = current_block.block_bbox
                next_bbox = next_block.block_bbox
                
                # 1. 空间逻辑优化：动态计算行高
                curr_height = max(1, curr_bbox[3] - curr_bbox[1])
                # 如果当前块包含多行，取平均行高（简单估算）
                line_count = max(1, len(current_block.block_content) // 40) # 假设每行约40字
                avg_line_height = curr_height / line_count
                
                vertical_gap = next_bbox[1] - curr_bbox[3]
                
                # 计算水平重叠度
                curr_x1, curr_x2 = curr_bbox[0], curr_bbox[2]
                next_x1, next_x2 = next_bbox[0], next_bbox[2]
                overlap_x1 = max(curr_x1, next_x1)
                overlap_x2 = min(curr_x2, next_x2)
                overlap_width = max(0, overlap_x2 - overlap_x1)
                curr_width = max(1, curr_x2 - curr_x1)
                x_alignment = overlap_width / curr_width
                
                # 2. 结构启发式：检查末尾标点
                ends_with_punc = bool(sentence_ends.search(current_block.block_content.strip()))
                
                # 合并决策逻辑：
                # 基础条件：水平对齐度高 (> 60%)
                # 空间条件：垂直间距 < 1.5 倍平均行高
                # 结构启发：如果末尾没有标点，且垂直间距在合理范围内 (如 3 倍行高)，也强制合并
                is_spatial_near = 0 <= vertical_gap < (avg_line_height * 1.5)
                is_structural_linked = not ends_with_punc and (0 <= vertical_gap < (avg_line_height * 3.0))
                
                if x_alignment > 0.6 and (is_spatial_near or is_structural_linked):
                    # 执行合并
                    connector = "" if not current_block.block_content.endswith("-") else ""
                    current_block.block_content += connector + " " + next_block.block_content
                    # 更新 bbox
                    current_block.block_bbox = [
                        min(curr_bbox[0], next_bbox[0]),
                        min(curr_bbox[1], next_bbox[1]),
                        max(curr_bbox[2], next_bbox[2]),
                        max(curr_bbox[3], next_bbox[3])
                    ]
                    continue
            
            merged.append(current_block)
            current_block = next_block
            
        merged.append(current_block)
        return merged

    async def query(
        self,
        doc_id: str,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        查询文档
        """
        if not self._initialized:
            await self.initialize()

        try:
            collection = self.collections.get(doc_id)
            if not collection:
                collection_name = f"doc_{doc_id}"
                try:
                    loop = asyncio.get_event_loop()
                    collection = await loop.run_in_executor(
                        None,
                        self.chroma_client.get_collection,
                        collection_name
                    )
                    self.collections[doc_id] = collection
                except Exception:
                    logger.warning(f"Collection for doc {doc_id} not found")
                    return []

            # 生成查询 embedding
            target_dim = self.collection_dims.get(doc_id)
            if not target_dim:
                try:
                    sample = collection.get(limit=1, include=["embeddings"])
                    if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                        target_dim = len(sample["embeddings"][0])
                except Exception:
                    target_dim = None
            if not target_dim:
                target_dim = 1024

            query_embeddings = await self._generate_embeddings([query_text], target_dim=target_dim)
            query_embedding = query_embeddings[0]

            # 查询 ChromaDB
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            )

            # 转换结果格式
            retrieved_results = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    metadata = results["metadatas"][0][i]
                    
                    # --- 优化召回逻辑: 如果是子块，优先提供完整的父块内容给 LLM ---
                    content = results["documents"][0][i]
                    if metadata.get("is_child") and metadata.get("full_block_content"):
                        content = metadata.get("full_block_content")
                    
                    result_item = {
                        "content": content,
                        "metadata": metadata,
                        "score": 1 - results["distances"][0][i],  # 距离转相似度
                    }
                    retrieved_results.append(result_item)

            return retrieved_results

        except Exception as e:
            logger.error(f"Error querying document {doc_id}: {e}")
            raise

    def _get_or_create_collection(self, collection_name: str):
        """获取或创建 collection"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists, clearing...")
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection

    def _get_block_type(self, label: str) -> str:
        """获取块类型的统一标签"""
        label = label.lower()

        if "table" in label:
            return "table"
        elif "formula" in label or "equation" in label:
            return "formula"
        elif "title" in label or "caption" in label or "footnote" in label:
            return "text"
        elif any(x in label for x in ["image", "figure", "chart"]):
            return "image"
        else:
            return "text"

    def format_citations(self, results: List[Dict[str, Any]]) -> List[Citation]:
        """
        将检索结果转换为前端引用格式
        """
        citations = []
        for i, result in enumerate(results):
            metadata = result["metadata"]
            content = result["content"]
            block_type = metadata.get("block_type", "text")

            try:
                bbox = json.loads(metadata.get("block_bbox", "[]"))
            except:
                bbox = None

            image_path = None
            block_label = metadata.get("block_label", "")
            doc_id = metadata.get("doc_id")

            if bbox and len(bbox) == 4 and block_label.lower() == "image":
                image_filename = f"img_in_image_box_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"
                image_path = f"/api/documents/{doc_id}/images/{image_filename}"

            citation = Citation(
                id=i + 1,
                source=metadata.get("file_name", ""),
                page=metadata.get("page_index", 0),
                snippet=content[:200],
                content=content,
                type=block_type,
                block_id=metadata.get("block_id"),
                bbox=bbox,
                image_path=image_path,
                score=result.get("score", 0.0)
            )
            citations.append(citation)

        return citations

    async def _generate_embeddings(self, texts: List[str], target_dim: Optional[int] = None) -> List[List[float]]:
        """
        使用 Qwen Embedding API 生成向量
        """
        if not settings.dashscope_api_key:
            dims = target_dim or 1024
            res = []
            for t in texts:
                res.append(self._fallback_embedding_for_text(t or "", dims))
            return res
        all_embeddings = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._call_embedding_api,
                    batch
                )
                if response.status_code == 200:
                    embeddings = [item['embedding'] for item in response.output['embeddings']]
                    if target_dim:
                        embeddings = [self._normalize_embedding_dim(e, target_dim) for e in embeddings]
                    all_embeddings.extend(embeddings)
                else:
                    dims = target_dim or 1024
                    for t in batch:
                        all_embeddings.append(self._fallback_embedding_for_text(t or "", dims))
            except Exception:
                dims = target_dim or 1024
                for t in batch:
                    all_embeddings.append(self._fallback_embedding_for_text(t or "", dims))
        return all_embeddings

    def _call_embedding_api(self, texts: List[str]):
        """同步调用 Embedding API"""
        response = TextEmbedding.call(
            model=settings.embedding_model,
            input=texts
        )
        return response

    def _fallback_embedding_for_text(self, text: str, dim: int) -> List[float]:
        data = text.encode("utf-8")
        out: List[float] = []
        counter = 0
        while len(out) < dim:
            h = hashlib.sha256(data + str(counter).encode("utf-8")).digest()
            for b in h:
                out.append(b / 255.0)
                if len(out) >= dim:
                    break
            counter += 1
        return out

    def _normalize_embedding_dim(self, vec: List[float], target_dim: int) -> List[float]:
        n = len(vec)
        if n == target_dim:
            return vec
        if n > target_dim:
            return vec[:target_dim]
        return vec + [0.0] * (target_dim - n)

    async def delete_document(self, doc_id: str):
        """删除文档索引"""
        try:
            collection_name = f"doc_{doc_id}"
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.chroma_client.delete_collection,
                collection_name
            )

            if doc_id in self.collections:
                del self.collections[doc_id]

            logger.info(f"Document {doc_id} deleted from index")
        except Exception as e:
            logger.warning(f"Error deleting document {doc_id}: {e}")

    async def cleanup(self):
        """清理资源"""
        self.collections.clear()
        self.chroma_client = None
        self._initialized = False
        logger.info("RAG service cleaned up")
