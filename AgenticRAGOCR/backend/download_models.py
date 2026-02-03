
import os
import sys
from pathlib import Path

def download_paddle_models():
    """使用 paddleocr 自带的下载机制触发模型下载"""
    print("正在准备下载 PaddleOCR 模型...")
    try:
        from paddleocr import PaddleOCR, PPStructure
        
        # 1. 触发基础 OCR 模型下载
        print("正在下载基础 OCR 模型 (ch_PP-OCRv4)...")
        PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        
        # 2. 触发布局检测与表格识别模型下载
        print("正在下载布局检测与版面恢复模型 (PP-Structure)...")
        PPStructure(recovery=True, show_log=False)
        
        print("\n模型下载完成！模型通常保存在 ~/.paddleocr/ 目录下。")
        print("如果您需要将模型移动到项目 backend/models 目录，请手动复制。")
        
    except ImportError:
        print("错误: 未安装 paddleocr。请先运行 pip install paddleocr paddlepaddle-gpu (或 paddlepaddle)")
    except Exception as e:
        print(f"下载过程中出现错误: {e}")

if __name__ == "__main__":
    download_paddle_models()
