#!/bin/bash

# 激活 conda 环境并启动后端
source /root/anaconda3/etc/profile.d/conda.sh
conda activate ocr_rag

# 获取端口配置
PORT=${PORT:-8100}

echo "Starting backend on port $PORT with ocr_rag conda environment..."
python start_backend.py
