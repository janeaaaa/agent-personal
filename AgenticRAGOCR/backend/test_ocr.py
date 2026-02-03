import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

# 添加 app 目录到路径
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.ocr_service import OCRService
from app.config import settings

async def test_ocr():
    # 实例化 OCRService
    ocr_service = OCRService()
    await ocr_service.initialize()
    # 确保 API Key 已设置
    if not settings.dashscope_api_key:
        print("Error: DASHSCOPE_API_KEY not found in settings")
        return

    # 测试文件路径
    test_pdf = os.path.join("uploads", "1.pdf")
    if not os.path.exists(test_pdf):
        print(f"Error: Test file not found at {test_pdf}")
        return

    print(f"Starting OCR test for: {test_pdf}")
    print(f"Using model: {settings.qwen_vl_model_name}")

    try:
        # 改进：我们直接调用 _dashscope_parse 的内部逻辑，但只传第一页，方便调试
        print("Extracting first page image...")
        import fitz
        from pathlib import Path
        doc = fitz.open(test_pdf)
        # 为了测试，我们只用前 2 页
        # 创建一个临时的 2 页 PDF 用于测试
        temp_dir = Path("temp_test")
        temp_dir.mkdir(exist_ok=True)
        temp_pdf_path = temp_dir / "test_2pages.pdf"
        
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=0, to_page=1)
        new_doc.save(str(temp_pdf_path))
        new_doc.close()
        doc.close()
        
        print(f"Created temp 2-page PDF at {temp_pdf_path}")

        # 调用 ocr_service.parse_document
        results = await ocr_service.parse_document(
            file_path=str(temp_pdf_path),
            doc_id="test_doc_single",
            output_dir="uploads/test_doc_single"
        )
        
        # results 是 List[OCRResult]，每一项对应一页
        print(f"\nSuccessfully parsed {len(results)} pages")
        
        for page_idx, page_res in enumerate(results):
            print(f"\n--- Page {page_idx + 1} ---")
            for i, block in enumerate(page_res.blocks):
                print(f"\nBlock {i+1}:")
                print(f"  Label: {block.block_label}")
                print(f"  BBox: {block.block_bbox}")
                print(f"  Image Path: {block.image_path}")
                # 打印内容的前 100 个字符
                content_preview = block.block_content[:100].replace('\n', ' ')
                print(f"  Content: {content_preview}...")
                
                # 检查图片是否成功生成
                if block.image_path:
                    # 解析 image_path 获取文件名
                    img_filename = os.path.basename(block.image_path)
                    # ocr_service.parse_document 会把裁剪后的图片放在 output_dir 中
                    abs_image_path = os.path.join("uploads/test_doc_single", img_filename)
                    
                    if os.path.exists(abs_image_path):
                        print(f"  [OK] Image file exists: {abs_image_path}")
                    else:
                        print(f"  [WARN] Image file NOT found at: {abs_image_path}")

    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ocr())
