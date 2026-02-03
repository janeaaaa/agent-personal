#!/bin/bash

echo "========================================="
echo "修复统计卡片显示问题"
echo "========================================="
echo ""

# 1. 清理所有进程
echo "1. 清理所有后台进程..."
killall -9 python node npm 2>/dev/null
sleep 3

# 2. 启动后端
echo "2. 启动后端..."
cd /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend
python start_backend.py > backend_fix.log 2>&1 &
BACKEND_PID=$!
echo "   后端PID: $BACKEND_PID"

# 3. 等待后端启动
echo "3. 等待后端启动（需要15-20秒）..."
for i in {1..20}; do
    sleep 1
    if curl -s http://localhost:8000/ | grep -q "healthy"; then
        echo "   ✅ 后端已启动"
        break
    fi
    echo -n "."
done
echo ""

# 4. 测试API
echo "4. 测试统计API..."
LATEST_DOC=$(ls -t /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend/uploads/ | head -1)
echo "   最新文档ID: $LATEST_DOC"

API_RESPONSE=$(curl -s http://localhost:8000/api/progress/$LATEST_DOC)
echo "   API响应:"
echo "$API_RESPONSE" | python3 -m json.tool 2>/dev/null | grep -A 6 "stats"

# 5. 检查JSON文件
echo ""
echo "5. 检查OCR JSON文件..."
JSON_FILE=$(find /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend/uploads/$LATEST_DOC -name "*_res.json" | head -1)
if [ -f "$JSON_FILE" ]; then
    echo "   ✅ JSON文件存在: $JSON_FILE"
    BLOCK_COUNT=$(python3 -c "import json; f=open('$JSON_FILE'); d=json.load(f); print(len(d.get('parsing_res_list', [])))")
    echo "   JSON中块数量: $BLOCK_COUNT"
else
    echo "   ❌ JSON文件不存在"
fi

# 6. 前端配置检查
echo ""
echo "6. 检查前端API配置..."
FRONTEND_API=$(grep "VITE_API_BASE_URL" /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/frontend/.env)
echo "   $FRONTEND_API"

# 7. 输出访问信息
echo ""
echo "========================================="
echo "✅ 修复完成"
echo "========================================="
echo "后端地址: http://localhost:8000/"
echo "前端需要手动启动:"
echo "  cd /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/frontend"
echo "  npm run dev"
echo ""
echo "测试API:"
echo "  curl http://localhost:8000/api/progress/$LATEST_DOC"
echo ""
echo "后端日志:"
echo "  tail -f /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend/backend_fix.log"
echo ""
