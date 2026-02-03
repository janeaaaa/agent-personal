# 多模态文档解析智能体 Frontend

基于 React + TypeScript 的 OCR RAG 系统前端界面，提供文档上传、索引构建和智能问答功能。

## 项目概述

这是一个现代化的前端应用，用于展示多模态文档理解和检索增强生成（RAG）系统。该系统支持 PDF、图片等多种格式文档的上传、解析、索引构建，并提供基于文档内容的智能问答功能。

## 技术栈

- **框架**: React 18 + TypeScript
- **构建工具**: Vite 5
- **样式**: Tailwind CSS 3
- **UI 组件库**: Shadcn/ui (基于 Radix UI)
- **动画**: Framer Motion
- **图标**: Lucide React
- **状态管理**: React Hooks

## 功能特性

### 文档上传与索引
- 拖拽式文件上传
- 支持 PDF、PNG、JPG 格式
- 实时上传进度显示
- 多阶段索引构建流程（上传 → 解析 → 嵌入 → 完成）
- 文档预览（原始 PDF 和解析布局对比）
- 提取内容统计（文本、表格、公式、图像）

### 智能问答
- 实时对话界面
- 带引用追踪的回答
- 可展开的引用详情
- 支持查看引用来源和上下文
- 多种内容类型标注（文本、表格、图像）

### UI 设计
- 深色渐变主题
- 流畅的动画效果
- 响应式布局
- 现代化的玻璃拟态设计

## 快速开始

### 环境要求

- Node.js >= 18.0.0
- npm >= 9.0.0 或 yarn >= 1.22.0

### 安装依赖

```bash
npm install
# 或
yarn install
```

### 启动开发服务器

```bash
npm run dev
# 或
yarn dev
```

应用将在 `http://localhost:3000` 启动。

### 构建生产版本

```bash
npm run build
# 或
yarn build
```

构建产物将输出到 `dist/` 目录。

### 预览生产构建

```bash
npm run preview
# 或
yarn preview
```

## 项目结构

```
frontend/
├── src/
│   ├── components/          # React 组件
│   │   ├── ui/             # Shadcn/ui 基础组件
│   │   ├── Header.tsx      # 顶部导航栏
│   │   ├── UploadPanel.tsx # 上传面板容器
│   │   ├── UploadZone.tsx  # 文件上传区域
│   │   ├── IndexingProgress.tsx  # 索引进度显示
│   │   ├── DocumentPreview.tsx   # 文档预览
│   │   ├── StatsCard.tsx   # 统计卡片
│   │   ├── ChatPanel.tsx   # 聊天面板容器
│   │   └── ChatMessage.tsx # 聊天消息组件
│   ├── lib/
│   │   └── utils.ts        # 工具函数
│   ├── styles/
│   │   └── globals.css     # 全局样式
│   ├── App.tsx             # 应用主组件
│   └── main.tsx            # 应用入口
├── index.html              # HTML 模板
├── package.json            # 项目配置
├── tsconfig.json           # TypeScript 配置
├── vite.config.ts          # Vite 配置
├── tailwind.config.js      # Tailwind 配置
└── postcss.config.js       # PostCSS 配置
```

## 后端接口对接

当前版本使用模拟数据进行演示。要对接 FastAPI 后端，需要在以下位置添加 API 调用：

### 上传文件
在 `UploadZone.tsx` 的 `simulateUpload` 函数中添加实际的上传逻辑：

```typescript
const handleFileUpload = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://your-api-url/upload', {
    method: 'POST',
    body: formData,
  });

  return await response.json();
};
```

### 问答请求
在 `ChatPanel.tsx` 的 `handleSend` 函数中添加 API 调用：

```typescript
const sendMessage = async (message: string) => {
  const response = await fetch('http://your-api-url/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  return await response.json();
};
```

## 自定义配置

### 主题颜色
在 `src/styles/globals.css` 中修改 CSS 变量来自定义主题：

```css
:root {
  --primary: #4A00E0;      /* 主色 */
  --accent: #00D9FF;       /* 强调色 */
  --background: #1E1E2F;   /* 背景色 */
  /* ... 更多颜色配置 */
}
```

### 端口配置
在 `vite.config.ts` 中修改开发服务器端口：

```typescript
export default defineConfig({
  server: {
    port: 3000,  // 修改为你想要的端口
  },
})
```

## 浏览器支持

- Chrome >= 90
- Firefox >= 88
- Safari >= 14
- Edge >= 90

## 许可证

本项目使用 MIT 许可证。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过项目 Issue 页面联系我们。
