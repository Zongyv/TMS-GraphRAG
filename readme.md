# rTMS-RAG: 重复经颅磁刺激知识图谱与Meta分析系统

基于 [nano-graphrag](https://github.com/gusye1234/nano-graphrag) 构建的多层连接研究知识图谱系统，专门用于 **重复经颅磁刺激（rTMS）** 领域的文献分析和Meta分析。

## 核心功能

- **关键词增强的GraphRAG**：在传统图RAG基础上加入关键词索引和语义索引，实现更精准的跨论文知识检索
- **多层次知识图谱**：支持从论文→实体→社区→全局的多层次查询
- **Meta分析集成**：自动提取效应量、森林图、偏倚风险评估等
- **多API支持**：兼容 OpenAI 协议的各种 LLM API（通义千问、DeepSeek、OpenAI等）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

复制环境变量模板并填入你的密钥：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入你的 API Key 和 Endpoint。

或者直接设置环境变量：

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
export LLM_MODEL=deepseek-v3
```

### 3. 准备数据

将你的 PDF/Markdown 论文放入 `process/dataset/rTMS/markdown/` 目录。

### 4. 运行

```bash
python test.py
```

## 项目结构

```
├── nano_graphrag/                   # 核心包
│   ├── keyword_based_graphrag.py    # 关键词增强的GraphRAG（核心）
│   ├── meta_analysis_graphrag.py    # Meta分析图RAG
│   ├── graphrag.py                  # 基础GraphRAG实现
│   ├── base.py                      # 基类定义
│   ├── prompt.py                    # Prompt模板
│   ├── _llm.py                      # LLM接口
│   ├── _op.py                       # 操作层
│   ├── _utils.py                    # 工具函数
│   ├── _storage/                    # 存储后端
│   └── entity_extraction/           # 实体提取模块
├── test.py                          # 主入口脚本
├── meta_analysis.py                 # Meta分析逻辑
├── outcome_aliases_config.py        # 结局指标配置
├── .env.example                     # 环境变量模板
└── requirements.txt                 # 依赖列表
```

## 许可证

MIT License

- 本项目的 GraphRAG 核心基于 [nano-graphrag](https://github.com/gusye1234/nano-graphrag) (MIT License, Copyright (c) 2024 Gustavo Ye) 重构
- 关键词增强、Meta分析集成等功能为独立新增