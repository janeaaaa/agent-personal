
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_qwen_vl_image():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    model = "qwen3-vl-plus-2025-12-19"
    image_path = "temp_test/test_p0.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing model: {model} with image: {image_path}...")
    base64_image = encode_image(image_path)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": "请识别并提取这张图片中的文字内容，以 JSON 格式输出：[{\"label\": \"text\", \"content\": \"...\"}]"},
                    ],
                }
            ],
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 2048
            }
        )
        
        print("\nResponse:")
        if hasattr(completion.choices[0].message, 'reasoning_content'):
            print(f"Reasoning: {completion.choices[0].message.reasoning_content}")
        print(f"Content: {completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_qwen_vl_image()
