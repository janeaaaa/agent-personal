import os
import asyncio
from app.services.llm_service import LLMService
from app.config import settings

async def test_llm():
    # 强制设置 API Key 用于测试
    api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-aa7f5f803e6c4508b6683ce4384139f4"
    os.environ["DASHSCOPE_API_KEY"] = api_key
    
    service = LLMService()
    await service.initialize()
    
    query = "如果 1+1=3，那么 2+2 等于多少？请详细推理。"
    context_chunks = [
        {
            "content": "在一个虚构的数学体系中，1+1 被定义为 3。",
            "metadata": {"block_type": "text", "page_index": 1}
        }
    ]
    
    print(f"Testing LLM with query: {query}")
    try:
        answer = await service.generate_answer(query, context_chunks, citations=[])
        print("\n--- Response ---")
        print(f"Answer:\n{answer}")
        print("----------------")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"LLM generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm())
