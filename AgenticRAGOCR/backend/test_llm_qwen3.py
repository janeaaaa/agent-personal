
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

async def test_qwen3_max():
    load_dotenv(override=True)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    print(f"Testing qwen3-max-2026-01-23 with api_key: {api_key[:10]}...")
    
    try:
        response = await client.chat.completions.create(
            model="qwen3-max-2026-01-23",
            messages=[
                {"role": "user", "content": "你好，请简单介绍下你自己。"}
            ],
            extra_body={"enable_thinking": True},
            stream=True
        )
        
        print("Response received:")
        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    print(f"Thinking: {delta.reasoning_content}")
                if hasattr(delta, "content") and delta.content:
                    print(f"Content: {delta.content}", end="", flush=True)
        print("\nTest completed successfully.")
        
    except Exception as e:
        print(f"Error testing qwen3-max-2026-01-23: {e}")
        
        print("Retrying without enable_thinking...")
        try:
            response = await client.chat.completions.create(
                model="qwen3-max-2026-01-23",
                messages=[
                    {"role": "user", "content": "你好，请简单介绍下你自己。"}
                ],
                stream=True
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\nRetry completed successfully.")
        except Exception as e2:
            print(f"Error even without enable_thinking: {e2}")

if __name__ == "__main__":
    asyncio.run(test_qwen3_max())
