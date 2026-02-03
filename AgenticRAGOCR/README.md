# 项目文档：Agentic RAG OCR

- 项目名：Agentic RAG OCR（高精度多模态文档解析与检索问答）
- 技术栈：FastAPI、PaddleOCR（可选，失败自动降级）、PyPDF2/pdfminer 回退解析、ChromaDB、DashScope Qwen、React + TypeScript

## 项目介绍
- 面向 PDF/图片类文档的解析、结构化、向量化与检索问答
- 通过 OCR 与布局分析抽取文档块（文本/表格/图片/公式），存储与索引，支持引文标注的问答输出
- 提供端到端上传、进度跟踪、块可视化、流式回答等前后端能力

## 技术架构
- 后端
  - API 层：FastAPI（上传、重处理、进度、块、可视化、查询/流式查询）
  - 解析层：PaddleOCRVL/PaddleOCR（可选）→ 回退文本解析（PyPDF2/pdfminer）
  - 检索层：文本分块、嵌入生成（Qwen Embedding）、ChromaDB 持久化
  - 进度持久化：每文档 progress.json，本地重启可恢复状态
- 前端
  - React + TypeScript，统一使用 VITE_API_BASE_URL 连接后端
  - 业务组件（上传、进度、预览、聊天）+ UI 基础组件库

## 实现流程
- 上传：保存文件至 uploads/<doc_id>；初始化进度（processing, 0%）
- 解析：优先 OCR 管线，失败自动降级至文本解析；生成结构化块与中间产物
- 统计：按块类型统计 text/table/image/formula
- 索引：分块、嵌入、入库（ChromaDB）；完成后进度标记 completed, 100%
- 查询：基于向量检索返回上下文，调用 Qwen 生成文本，返回含引文的结果
- 可视化：提供原图、布局检测图、阅读顺序图等文件访问

## 技术方案
- OCR 可选化：PaddleOCRVL/PaddleOCR 初始化失败时自动回退；保持解析链路可用
- 文本回退解析：PyPDF2 优先；若无有效文本，尝试 pdfminer；统一做 UTF-8 规范化与控制字符清理
- 文本回退解析：PyPDF2 优先；若无有效文本，尝试 pdfminer；统一做 UTF-8 规范化与控制字符清理
- EasyOCR 回退：图片类文件使用 EasyOCR 进行文本识别并输出 bbox，提高图片解析的可用性（PDF 仍走文本回退）
- 检索与嵌入：RecursiveCharacterTextSplitter；Qwen Embedding 批量；ChromaDB PersistentClient（禁用遥测）
- 统一状态存储：引入 SQLite（progress.db）保存进度与任务尝试次数，服务重启后可恢复；失败自动重试（最多2次）
- 检索与嵌入：RecursiveCharacterTextSplitter；Qwen Embedding 批量；ChromaDB PersistentClient（禁用遥测）
- 索引前文本增强：NFC 归一化、空白与换行清理、段落与连字处理，提高分块与检索颗粒度
- 前端统一基址：通过环境变量 VITE_API_BASE_URL，避免端口硬编码导致连接问题

## 可实现功能
- 文档上传与自动解析、索引构建
- 进度查询与统计展示（页块数、类型分布）
- 按类型/页码获取块内容，含元数据（源文件、bbox、图片路径）
- 向量检索与问答，支持流式输出与引文标注
- 文档重处理（重启后恢复）

## 技术边界
- OCR 能力依赖环境与模型版本；当前在不可用时降级为纯文本解析，表格/图片/公式识别能力受限
- pdfminer 在复杂版式下页边界保留不稳定；分页使用近似策略
- 嵌入与检索质量受文本清洗与分块策略影响；大文档需关注性能与资源
- Qwen 生成受 API 限制（批量大小、速率），网络异常会触发降级回答

## 未来迭代方向与计划
- 视觉能力恢复与增强：修复 Paddle 版本兼容，加入表格/图片/公式专用识别与结构化
- 视觉能力恢复与增强：修复 Paddle 版本兼容，加入表格/图片/公式专用识别与结构化；探索对 PDF 页面的图像化识别通道
- 文本质量提升：更细粒度的编码清洗、版式保留与段落结构识别
- 评测框架：加入端到端集成测试与数据基准，量化检索与生成效果
- 评测框架：加入端到端集成测试与数据基准，量化检索与生成效果（召回率、准确率、延迟等）


## 使用说明
- 后端：`python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
- 前端：设置 `VITE_API_BASE_URL=http://localhost:8000` 后正常构建与运行
- API 主要路径：
  - POST /api/upload
  - POST /api/upload
  - POST /api/documents/{doc_id}/reprocess
  - GET /api/progress/{doc_id}
  - GET /api/documents/{doc_id}/blocks
  - GET /api/documents/{doc_id}/visualizations
  - POST /api/query、POST /api/query/stream

## 配置与环境
### 1. API 密钥配置 (关键)
本项目使用阿里云百炼平台 (DashScope) 的模型能力。为了安全起见，**请勿将 API 密钥直接硬编码在代码中**。
- **配置方法**：
  1. 在 `AgenticRAGOCR/backend/` 目录下创建 `.env` 文件（可参考 `.env.example`）。
  2. 在 `.env` 中添加：`DASHSCOPE_API_KEY=你的密钥`。
  3. 也可以通过系统环境变量设置 `DASHSCOPE_API_KEY`。
- **注意**：`.env` 文件已被包含在 `.gitignore` 中，不会被提交到仓库。每次 fork 或克隆项目后，请务必手动配置。

### 2. 依赖项
- **后端**：FastAPI、chromadb、dashscope、langchain、pysqlite3
- **解析**：paddleocr（可选）、PyPDF2、pdfminer.six、easyocr
- **前端**：React、Vite、Tailwind CSS

## 测试与评估（建议）
- 集成测试：上传→解析→进度→块→检索→问答 全链路脚本
- 评估指标：检索召回率、回答准确率、响应延迟、解析耗时、降级触发率
 
 ## Redis 状态存储（可选）
 - 通过环境变量启用：`USE_REDIS=true`、`REDIS_URL=redis://localhost:6379/0`
 - 优先级：Redis > SQLite（progress.db）> 内存，服务重启后可恢复进度与任务尝试次数
 - 适用于并发与分布式场景；本地单机默认使用 SQLite 持久化
 - 配置项位置：[config.py](file:///f:/BaiduNetdiskDownload/15套原创热门Agent项目源码/双十二/高精度多模态文档解析/AgenticRAGOCR/backend/app/config.py#L40-L43)
 
 ## PPStructure 视觉块识别
 - 依赖 PaddleOCR 的 PPStructure，支持表格/图片/公式等结构化识别
 - 当前解析链路：PPStructure（图片）→ EasyOCR（回退）→ 文本回退（PyPDF2/pdfminer）
 - 模型路径配置项：
   - `paddleocr_vl_model_dir`
   - `layout_detection_model_dir`
 - 对应实现位置：[ocr_service.py](file:///f:/BaiduNetdiskDownload/15套原创热门Agent项目源码/双十二/高精度多模态文档解析/AgenticRAGOCR/backend/app/services/ocr_service.py#L513-L521)
 
 ## 集成测试执行
 - 启动后端服务：`python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
 - 准备测试数据：在 `tests/data/` 放置 `sample.pdf`
 - 运行脚本：`python tests/run_e2e.py`（可选设置 `API_BASE_URL`）
 - 预期结果：进度达到 completed/100%，块列表非空
 - 脚本位置与说明：
   - 测试脚本：[run_e2e.py](file:///f:/BaiduNetdiskDownload/15套原创热门Agent项目源码/双十二/高精度多模态文档解析/AgenticRAGOCR/tests/run_e2e.py)
   - 数据说明：[tests/data/README.txt](file:///f:/BaiduNetdiskDownload/15套原创热门Agent项目源码/双十二/高精度多模态文档解析/AgenticRAGOCR/tests/data/README.txt)
