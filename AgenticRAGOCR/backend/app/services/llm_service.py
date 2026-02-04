"""
LLM Service for Alibaba Cloud DashScope (Qwen)
"""

import logging
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator

import dashscope
from dashscope import Generation

from app.config import settings
from app.models.schemas import Citation

logger = logging.getLogger(__name__)

class LLMService:
    """阿里云百炼平台 Qwen 模型服务 (恢复原始版本并保留异步接口)"""

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """初始化 LLM 服务"""
        if self._initialized:
            return

        try:
            logger.info("Initializing LLM service...")

            # 优先使用 settings 中的配置
            api_key = settings.dashscope_api_key
            if not api_key:
                api_key = os.getenv("DASHSCOPE_API_KEY")
            
            if not api_key:
                logger.warning("DASHSCOPE_API_KEY is not set. LLM service might fail.")
            
            # 设置 DashScope API Key
            dashscope.api_key = api_key

            self._initialized = True
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> str:
        """
        基于检索上下文生成回答

        Args:
            query: 用户查询
            context_chunks: 检索到的上下文块
            citations: 引用列表

        Returns:
            生成的回答文本
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 构建上下文
            context_parts = []
            
            # 优先提取全局总结
            global_summary = ""
            for chunk in context_chunks:
                if chunk["metadata"].get("is_summary"):
                    global_summary = chunk["content"]
                    break
            
            if global_summary:
                context_parts.append(f"【文档全局背景】\n{global_summary}")

            for i, chunk in enumerate(context_chunks):
                if chunk["metadata"].get("is_summary"):
                    continue # 已经添加过了
                
                metadata = chunk["metadata"]
                content = chunk["content"]
                block_type = metadata.get("block_type", "text")
                page = metadata.get("page_index", 0)

                # 根据块类型添加前缀
                if block_type == "table":
                    prefix = f"[表格 - 第{page}页]"
                elif block_type == "image":
                    prefix = f"[图像/图表 - 第{page}页]"
                elif block_type == "formula":
                    prefix = f"[公式 - 第{page}页]"
                else:
                    prefix = f"[文本 - 第{page}页]"

                context_parts.append(f"{prefix}\n{content}")

            context = "\n\n".join(context_parts)

            # 构建引用映射
            citation_map = {}
            for citation in citations:
                citation_map[citation.id] = f"【{citation.id}】"

            # 构建 prompt (增强约束：严禁发散，引用优先)
            system_prompt = """你是一个极其严谨的文档问答助手。你的任务是基于提供的文档上下文，提供事实性的回答。

必须遵守的铁律：
1. **严禁发散与猜测**：如果文档中没有明确提到的数字、日期、结论或任何事实，严禁根据常识、预训练知识或逻辑推导进行猜测。如果信息不足，请直接回答“文档中未提及此信息”。
2. **空间感知与临近性**：上下文可能包含 `[前文内容]`、`[当前命中的核心内容]` 和 `[后文内容]`。请综合这些临近信息来理解当前块的完整语义，特别是当核心内容是表格描述或公式推导的一部分时。
3. **引用优先**：你的每一个核心观点、事实、数据，都必须在句尾紧跟对应的引用编号（例如【1】【2】）。严禁提供任何无法溯源到上下文的断言。
4. **忠实原意**：不要试图美化或改变文档的原意。对于表格、图像、公式等内容，请明确指出其所在的页码和类型。
5. **简洁准确**：回答应结构清晰，避免冗长的废话。

引用标注规则：
- 使用【1】【2】【3】等数字标记，对应下方“文档上下文”中的顺序。
- 多个来源请连续标注，如【1】【2】。
"""

            user_prompt = f"""请严格基于以下“文档上下文”回答用户问题。如果上下文无法支持回答，请直接告知。

## 文档上下文
{context}

## 用户问题
{query}

请提供准确的回答，并确保每个事实点都有引用标注：
"""

            # 调用 Qwen 模型 (采用原始版本的 Generation.call 方式)
            logger.info(f"Calling Qwen model: {settings.qwen_model_name}")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_qwen_sync,
                system_prompt,
                user_prompt,
                False # stream=False
            )

            # 处理响应 (兼容不同版本的返回格式)
            answer = ''
            if hasattr(response, 'output'):
                if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                    choice = response.output.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        answer = choice.message.content
                elif hasattr(response.output, 'text'):
                    answer = response.output.text
                elif isinstance(response.output, dict):
                    answer = response.output.get('text', '')

            if answer is None:
                answer = ''

            logger.info(f"Generated answer: {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            parts = []
            for i, c in enumerate(citations):
                parts.append(f"【{i+1}】")
            prefix = "参考文档片段 " + "".join(parts) if parts else ""
            fallback = f"{prefix}\n问题：{query}\n回答基于检索到的上下文。"
            return fallback

    def _call_qwen_sync(self, system_prompt: str, user_prompt: str, stream: bool = False):
        """同步调用 Qwen 模型"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if stream:
            return Generation.call(
                model=settings.qwen_model_name,
                messages=messages,
                result_format='message',
                temperature=0.01, # 极低温度，确保输出高度确定，不发散
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
                stream=True,
                incremental_output=True
            )
        else:
            response = Generation.call(
                model=settings.qwen_model_name,
                messages=messages,
                result_format='message',
                temperature=0.01, # 极低温度
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
            )
            if response.status_code != 200:
                raise Exception(f"Qwen API error: {response.code} - {response.message}")
            return response

    async def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答（支持 SSE）
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 构建上下文
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                metadata = chunk["metadata"]
                content = chunk["content"]
                block_type = metadata.get("block_type", "text")
                page = metadata.get("page_index", 0)

                if block_type == "table":
                    prefix = f"[表格 - 第{page}页]"
                elif block_type == "image":
                    prefix = f"[图像/图表 - 第{page}页]"
                elif block_type == "formula":
                    prefix = f"[公式 - 第{page}页]"
                else:
                    prefix = f"[文本 - 第{page}页]"

                context_parts.append(f"{prefix}\n{content}")

            context = "\n\n".join(context_parts)

            system_prompt = """你是一个专业的文档问答助手。你的任务是：
1. 基于提供的文档上下文，准确回答用户的问题
2. 在回答中使用【数字】标记引用来源（例如【1】【2】）
3. 对于表格、图像、公式等特殊内容，明确指出其类型
4. 如果上下文中没有相关信息，诚实地说明
5. 回答要准确、简洁、结构清晰

引用标注规则：
- 使用【1】【2】【3】等数字标记，对应检索到的文档块
- 每个关键信息点都应该标注引用来源
- 多个来源可以连续标注，如【1】【2】
"""

            user_prompt = f"""基于以下文档上下文，回答用户的问题。

## 文档上下文
{context}

## 用户问题
{query}

请提供详细的回答，并在关键信息处标注引用来源【1】【2】等。
"""

            # 采用同步生成器包装为异步
            responses = self._call_qwen_sync(system_prompt, user_prompt, True)

            for response in responses:
                if response.status_code == 200:
                    chunk = ''
                    if hasattr(response, 'output'):
                        if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                            choice = response.output.choices[0]
                            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                                chunk = choice.message.content
                        elif hasattr(response.output, 'text'):
                            chunk = response.output.text
                        elif isinstance(response.output, dict):
                            chunk = response.output.get('text', '')
                    if chunk:
                        yield chunk
                    await asyncio.sleep(0) # 给其他协程机会
                else:
                    break
        except Exception as e:
            logger.error(f"Error in generate_answer_stream: {e}")
            yield "生成失败，返回基于检索内容的简要回答。"

    async def cleanup(self):
        """清理资源"""
        self._initialized = False
        logger.info("LLM service cleaned up")

    async def generate_summary(self, text: str, max_chars: int = 500) -> str:
        """
        为文本生成摘要 (用于增强 RAG 全局语义)
        """
        if not self._initialized:
            await self.initialize()

        if not text or len(text) < 100:
            return text

        try:
            system_prompt = f"你是一个专业的文档摘要助手。请用一两句话概括以下内容的重点，字数控制在 {max_chars} 字以内。要求简洁、准确，保留核心语义。"
            user_prompt = f"请概括以下内容：\n\n{text[:10000]}" # 限制输入长度，防止 token 溢出

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_qwen_sync,
                system_prompt,
                user_prompt,
                False
            )

            summary = ''
            if hasattr(response, 'output'):
                if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                    summary = response.output.choices[0].message.content
                elif hasattr(response.output, 'text'):
                    summary = response.output.text
            
            return summary.strip() or text[:max_chars]
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:max_chars]
