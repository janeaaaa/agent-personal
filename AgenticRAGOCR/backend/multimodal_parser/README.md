# Multimodal Parser

独立的多模态文档解析工具，基于阿里云 Qwen-VL 模型，支持 PDF 和 图片的高精度解析。

## 功能特点

- **多模态解析**：支持文本、表格、公式、图像等多种内容块的识别与提取。
- **高精度 OCR**：利用 Qwen-VL 的视觉能力，提供逐行、精准的文字识别。
- **结构化输出**：直接输出 JSON 格式的结构化数据，包含每个内容块的类型、内容、坐标位置 (bbox)。
- **图像自动裁剪**：自动从文档中裁剪出识别到的图像/图表，并保存为独立文件。
- **表格 Markdown 化**：自动将表格内容转换为 Markdown 格式，便于后续处理。

## 依赖安装

```bash
pip install openai pillow pymupdf requests
```

## 使用方法

### 命令行运行

```bash
# 设置 API Key (Linux/Mac)
export DASHSCOPE_API_KEY="your_api_key_here"

# 设置 API Key (Windows PowerShell)
$env:DASHSCOPE_API_KEY="your_api_key_here"

# 运行解析
python parser.py <文件路径>
```

或者直接在命令行中传入 API Key：

```bash
python parser.py <文件路径> <your_api_key_here>
```

### Python 代码调用

```python
from parser import MultimodalParser

# 初始化解析器
parser = MultimodalParser(api_key="your_api_key_here")

# 解析文件
results = parser.parse_file("path/to/document.pdf")

# 处理结果
for page in results:
    print(f"Page {page['page_index']} blocks: {len(page['parsing_res_list'])}")
```

## 输出结果

解析结果将保存在源文件同级目录下的 `*_result.json` 文件中，提取的图片将保存在 `*_parsed` 文件夹中。

JSON 结构示例：

```json
[
  {
    "page_index": 0,
    "parsing_res_list": [
      {
        "block_id": 0,
        "block_label": "text",
        "block_content": "文档标题...",
        "block_bbox": [100, 100, 900, 200],
        "block_order": 0
      },
      {
        "block_id": 1,
        "block_label": "image",
        "block_content": "图表描述...",
        "block_bbox": [100, 300, 500, 600],
        "image_save_path": ".../page_0_img_1.jpg"
      }
    ]
  }
]
```
