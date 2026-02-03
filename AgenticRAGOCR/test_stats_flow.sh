#!/bin/bash

echo "========== 测试统计卡片显示流程 =========="
echo ""

# 检查后端
echo "1. 检查后端服务 (端口 8100)..."
if curl -s http://localhost:8100/ > /dev/null 2>&1; then
    echo "   ✓ 后端服务运行正常"
else
    echo "   ✗ 后端服务未运行"
    exit 1
fi

echo ""

# 检查前端
echo "2. 检查前端服务 (端口 3000)..."
if curl -s http://localhost:3000/ > /dev/null 2>&1; then
    echo "   ✓ 前端服务运行正常"
else
    echo "   ✗ 前端服务未运行"
    exit 1
fi

echo ""

# 测试已存在文档的统计API
echo "3. 测试进度/统计API (使用已存在的文档)..."
DOC_ID="85ffeb42-5c20-41e3-b0ea-c9737caf838e"
echo "   文档ID: $DOC_ID"

STATS_RESPONSE=$(curl -s http://localhost:8100/api/progress/$DOC_ID)
echo ""
echo "   API响应:"
echo "$STATS_RESPONSE" | python3 -m json.tool

echo ""
echo "   提取统计信息:"
TEXT_BLOCKS=$(echo "$STATS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('stats', {}).get('text_blocks', 0))" 2>/dev/null)
TABLE_BLOCKS=$(echo "$STATS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('stats', {}).get('table_blocks', 0))" 2>/dev/null)
IMAGE_BLOCKS=$(echo "$STATS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('stats', {}).get('image_blocks', 0))" 2>/dev/null)
FORMULA_BLOCKS=$(echo "$STATS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('stats', {}).get('formula_blocks', 0))" 2>/dev/null)
TOTAL_BLOCKS=$(echo "$STATS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('stats', {}).get('total_blocks', 0))" 2>/dev/null)

echo "   - 文本块: $TEXT_BLOCKS"
echo "   - 表格块: $TABLE_BLOCKS"
echo "   - 图像块: $IMAGE_BLOCKS"
echo "   - 公式块: $FORMULA_BLOCKS"
echo "   - 总块数: $TOTAL_BLOCKS"

if [ "$TOTAL_BLOCKS" -gt 0 ]; then
    echo "   ✓ 统计数据正常"
else
    echo "   ✗ 统计数据异常"
fi

echo ""
echo "4. 检查OCR结果文件..."
JSON_FILE=$(find /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend/uploads/$DOC_ID -name "*_res.json" | head -1)
if [ -f "$JSON_FILE" ]; then
    echo "   ✓ JSON文件存在: $JSON_FILE"
    BLOCK_COUNT=$(python3 -c "import json; data=json.load(open('$JSON_FILE')); print(len(data.get('parsing_res_list', [])))")
    echo "   - JSON中的块数: $BLOCK_COUNT"
else
    echo "   ✗ JSON文件不存在"
fi

echo ""
echo "========== 测试完成 =========="
echo ""
echo "如果所有测试通过,前端应该能够正确显示统计卡片。"
echo "请在浏览器中访问: http://localhost:3000"
echo ""
