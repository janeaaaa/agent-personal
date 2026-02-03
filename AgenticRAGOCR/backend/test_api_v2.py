
import os
import dashscope
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv("DASHSCOPE_API_KEY")

print(f"Testing API Key: {api_key[:5]}...{api_key[-5:] if api_key else ''}")

def test_embedding():
    print("\n--- Testing Embedding ---")
    try:
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v3",
            input="Hello",
            api_key=api_key
        )
        if resp.status_code == 200:
            print("Embedding Success!")
        else:
            print(f"Embedding Failed: {resp.code} - {resp.message}")
    except Exception as e:
        print(f"Embedding Exception: {e}")

def test_chat():
    print("\n--- Testing Chat (OpenAI compatible) ---")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[{'role': 'user', 'content': 'Hi'}]
        )
        print(f"Chat Success: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"Chat Exception: {e}")

if __name__ == "__main__":
    test_embedding()
    test_chat()
