# 前端数据渲染测试指南

## 问题分析

目前的问题：
1. ✅ 文档解析成功（JSON文件已生成）
2. ✅ 统计数据正确：17文本 + 2表格 + 1图像 + 2公式 = 22总块
3. ❌ 但前端显示全是0（因为向量索引时统计信息未正确保存）

## 快速测试方法（无需重新解析）

### 方法1: 使用Mock API测试前端渲染

#### 1. 测试Mock端点
```bash
curl -s http://localhost:8000/api/test/mock-progress | jq .
```

返回完整的模拟数据：
```json
{
  "doc_id": "test-doc-id-12345",
  "status": "completed",
  "progress": 100,
  "message": "文档解析完成！",
  "stats": {
    "doc_id": "test-doc-id-12345",
    "text_blocks": 15,
    "table_blocks": 3,
    "image_blocks": 2,
    "formula_blocks": 4,
    "total_blocks": 24
  }
}
```

#### 2. 在浏览器控制台测试

打开 http://localhost:3002/ 并在浏览器Console中运行：

```javascript
// 测试API调用
fetch('http://localhost:8000/api/test/mock-progress')
  .then(r => r.json())
  .then(data => {
    console.log('Mock数据:', data);
    // 手动设置到Context中测试渲染
    // 这需要在React DevTools中操作
  });
```

#### 3. 临时修改前端代码测试

修改 `IndexingProgress.tsx` 中的API调用来使用mock端点：

```typescript
// 原代码（第XX行）:
const progressData = await getIndexingProgress(docId);

// 临时改为:
const progressData = await fetch('http://localhost:8000/api/test/mock-progress').then(r => r.json());
```

这样前端就会立即显示完整的统计数据，验证渲染逻辑是否正确。

### 方法2: 手动修复已解析文档的统计数据

创建一个修复脚本来重新计算并更新统计信息：

```bash
# 运行修复脚本
cd /home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend
python fix_stats.py
```

### 方法3: 直接curl测试每个endpoint

```bash
# 1. 测试健康检查
curl http://localhost:8000/

# 2. 测试Mock进度（有完整stats）
curl http://localhost:8000/api/test/mock-progress | jq .

# 3. 测试真实进度（当前stats全是0）
curl http://localhost:8000/api/progress/fee551f0-e8ad-4959-9a8f-541ef6f74f48 | jq .
```

## 真实问题的修复方案

### 问题根因

在 `app/services/rag_service.py` 的 `index_document` 方法中：
- ✅ 统计逻辑正确（`_get_block_type`方法正常）
- ❌ 但某个环节导致统计数据未正确保存到`stats`对象

### 需要检查的代码位置

文件: `backend/app/services/rag_service.py`

1. **统计初始化** (约第123行):
```python
stats = DocumentStats(
    doc_id=doc_id,
    text_blocks=0,
    table_blocks=0,
    image_blocks=0,
    formula_blocks=0,
    total_blocks=0
)
```

2. **统计更新逻辑** (约第136-147行):
```python
for block in ocr_result.blocks:
    stats.total_blocks += 1

    block_type = self._get_block_type(block.block_label)
    if block_type == "text":
        stats.text_blocks += 1
    elif block_type == "table":
        stats.table_blocks += 1
    elif block_type == "image":
        stats.image_blocks += 1
    elif block_type == "formula":
        stats.formula_blocks += 1
```

3. **类型映射** (约第296-307行):
```python
def _get_block_type(self, label: str) -> str:
    label = label.lower()

    if "table" in label:
        return "table"
    elif any(x in label for x in ["image", "figure", "chart"]):
        return "image"
    elif "formula" in label or "equation" in label:
        return "formula"
    else:
        return "text"
```

### 可能的问题

1. **OCR结果为空**: `ocr_results` 列表可能为空
2. **blocks为空**: `ocr_result.blocks` 可能为空
3. **数据格式不匹配**: 从JSON解析的数据结构与预期不符

## 当前服务状态

- ✅ 前端: http://localhost:3002/
- ✅ 后端: http://localhost:8000/
- ✅ Mock API: http://localhost:8000/api/test/mock-progress
- ⚠️  真实解析: 数据存在但统计为0

## 推荐测试流程

1. **验证前端渲染**:
   - 访问 http://localhost:3002/
   - 在浏览器Console运行fetch测试
   - 确认前端能正确显示mock数据

2. **定位统计bug**:
   - 添加日志到rag_service.py
   - 重新上传测试文件
   - 查看后端日志确认统计逻辑执行情况

3. **修复并验证**:
   - 修复统计逻辑bug
   - 重新上传文档
   - 确认前端显示正确数据
