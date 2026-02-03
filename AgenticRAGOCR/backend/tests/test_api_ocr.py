
import requests
import time
import os
import sys

def test_upload_and_parse(file_path):
    url = "http://localhost:8000"
    
    # 1. 上传文件
    print(f"正在上传文件: {file_path}")
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{url}/api/upload", files=files)
    
    if response.status_code != 200:
        print(f"上传失败: {response.status_code} - {response.text}")
        return
    
    data = response.json()
    doc_id = data["doc_id"]
    print(f"上传成功, doc_id: {doc_id}")
    
    # 2. 轮询状态
    print("正在等待解析完成...")
    while True:
        status_response = requests.get(f"{url}/api/progress/{doc_id}")
        if status_response.status_code != 200:
            print(f"获取进度失败: {status_response.status_code} - {status_response.text}")
            break
            
        status_data = status_response.json()
        progress = status_data.get("progress", 0)
        status = status_data.get("status", "unknown")
        
        print(f"当前进度: {progress}% (状态: {status})")
        
        if status == "completed":
            print("解析完成！")
            break
        elif status == "failed":
            print(f"解析失败: {status_data.get('message') or status_data.get('error')}")
            break
            
        time.sleep(2)
    
    # 3. 获取解析结果
    print("获取解析出的块...")
    blocks_response = requests.get(f"{url}/api/documents/{doc_id}/blocks")
    if blocks_response.status_code == 200:
        data = blocks_response.json()
        blocks = data.get("blocks", [])
        print(f"成功获取到 {len(blocks)} 个块")
        # 打印前 2 个块的内容作为验证
        for i, block in enumerate(blocks[:2]):
            print(f"块 {i+1} (类型: {block.get('block_label')}): {block.get('block_content')[:100]}...")
    else:
        print(f"获取块失败: {blocks_response.status_code} - {blocks_response.text}")

if __name__ == "__main__":
    # 使用 upload/1.pdf 进行测试
    test_file = "uploads/1.pdf"
    if not os.path.exists(test_file):
        test_file = os.path.join(os.path.dirname(__file__), "..", "uploads", "1.pdf")
    
    if os.path.exists(test_file):
        test_upload_and_parse(test_file)
    else:
        print(f"未找到测试文件: {test_file}")
