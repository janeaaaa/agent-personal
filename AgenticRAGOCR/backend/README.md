# 多模态文档解析智能体 Backend

多模态 RAG 系统后端服务，支持 PDF 和图片文档的解析、索引和问答。

## 技术栈

- **FastAPI**: REST API 框架
- **PaddleOCR-VL**: 多模态文档解析（文本、表格、图像、公式）
- **ChromaDB**: 向量数据库
- **LangChain**: RAG 编排
- **Alibaba Cloud DashScope**: Qwen 大语言模型 + Embedding 模型

## 功能特性

✅ 支持 PDF、PNG、JPG 等格式的文档解析
✅ **多层级 OCR 引擎**: 优先使用 PaddleOCRVL，支持 EasyOCR、PyPDF2、pdfminer.six 多级回退
✅ **DashScope API 模式**: 集成阿里云百炼 Qwen-VL，支持云端高精度多模态解析
✅ **增强文本清洗**: 深度 Unicode 清洗，移除不可见字符、控制字符及私有区字符 (PUA)
✅ **统一状态管理**: 引入 SQLite (任务状态) 与 Redis (任务调度)，支持并发处理与失败重试
✅ 自动识别文本、表格、图像、公式等多种内容类型
✅ 向量化索引，支持语义检索与完美的引用溯源
✅ 基于 Qwen 的智能问答，支持流式响应

## 安装步骤

### 1. 安装 Python 依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

推荐使用系统环境变量配置 API Key，避免硬编码：

**PowerShell (Windows):**
```powershell
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'your_api_key', 'User')
```

**Bash (Linux/macOS):**
```bash
echo "export DASHSCOPE_API_KEY='your_api_key'" >> ~/.bashrc
source ~/.bashrc
```

或者复制 `.env.example` 为 `.env` 并填写配置：

```env
# 阿里云百炼 API Key
DASHSCOPE_API_KEY=your_api_key_here

# OCR 后端选择: local (PaddleOCR) 或 dashscope (Qwen-VL API)
OCR_BACKEND=dashscope

# Redis 配置 (可选)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. 准备本地模型 (仅 Local 模式需要)

运行下载脚本自动下载并解压 PaddleOCR 相关模型：
```bash
python download_models.py
```

### 4. 启动服务

```bash
# 启动后端服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 测试与验证

项目包含集成测试脚本，可验证 API 解析流程：

```bash
# 运行 API 模式集成测试
python tests/test_api_ocr.py
```

解析结果保存在 `uploads/{doc_id}/` 目录下：
- `full_res.json`: 全量结构化解析结果
- `full_res.md`: 可视化 Markdown 文档内容

## 项目结构

```
backend/
├── app/
│   ├── main.py                 # FastAPI 应用与路由
│   ├── config.py               # 配置与环境变量管理
│   ├── models/                 # Pydantic 模式定义
│   └── services/
│       ├── ocr_service.py      # 多级 OCR 与 API 解析逻辑
│       ├── rag_service.py      # ChromaDB 向量索引与检索
│       └── llm_service.py      # Qwen 模型调用
├── tests/                      # 集成测试脚本
├── uploads/                    # 文档存储与解析结果输出
├── data/                       # 数据库与向量存储
├── .env                        # 环境配置
└── requirements.txt            # 依赖清单
```

## 开发说明

### 日志

日志会输出到控制台，包含以下信息：
- 服务启动/关闭
- 文档上传和处理进度
- OCR 解析结果
- 向量索引构建
- 查询请求和响应

### 数据持久化

- **上传文件**: `./uploads/{doc_id}/`
- **ChromaDB**: `./data/chroma_db/`
- **OCR 输出**: `./uploads/{doc_id}/*.json` 和 `*.md`

### 性能优化

- OCR 和 Embedding 在线程池中异步执行，避免阻塞
- 支持批量向量化
- ChromaDB 使用 HNSW 索引加速检索

## 常见问题

**Q: 如何获取阿里云百炼 API Key？**
A: 访问 https://dashscope.console.aliyun.com/ 注册并获取 API Key。

**Q: PaddleOCR 模型太大怎么办？**
A: 可以使用 PaddleOCR 提供的轻量级模型，或使用云端 OCR API。

**Q: 支持哪些文档格式？**
A: 目前支持 PDF、PNG、JPG、JPEG。可以通过修改 `settings.allowed_extensions` 扩展。

**Q: 如何调整检索数量？**
A: 在 `.env` 中设置 `RETRIEVAL_TOP_K=10`（默认 5）。

**Q: 使用的是哪个 Embedding 模型？**
A: 使用阿里云 DashScope 的 `text-embedding-v3` 模型，支持中英文，维度 1024，无需本地部署。

## License

MIT
