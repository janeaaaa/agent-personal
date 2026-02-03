
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
if "DASHSCOPE_API_KEY" in os.environ:
    del os.environ["DASHSCOPE_API_KEY"]
load_dotenv(override=True)

def test_qwen_vl_minimal():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found")
        return
    print(f"Key length: {len(api_key)}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    model = "qwen3-vl-flash-2026-01-22"
    
    # 使用一个简单的测试图片（如果存在）或跳过图片部分
    # 这里我们只测试纯文本，看 API 是否连通
    print(f"Testing model: {model} with minimal prompt...")
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "你好，请确认你是否能正常工作。"},
                    ],
                }
            ],
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 1024
            }
        )
        
        print("\nResponse:")
        if hasattr(completion.choices[0].message, 'reasoning_content'):
            print(f"Reasoning: {completion.choices[0].message.reasoning_content}")
        print(f"Content: {completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_qwen_vl_minimal()
