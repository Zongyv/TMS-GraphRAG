# TMS-GraphRAG

**Knowledge Graph & Meta-Analysis System for rTMS (Repetitive Transcranial Magnetic Stimulation)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-layered knowledge graph system built on [nano-graphrag](https://github.com/gusye1234/nano-graphrag), designed for literature analysis and meta-analysis in the **rTMS** domain.

---

[中文](#中文) • [English](#english)

---

## English

### Features

- **Keyword-enhanced GraphRAG**: Augments traditional GraphRAG with keyword indexing and semantic indexing for more precise cross-paper knowledge retrieval
- **Multi-layered Knowledge Graph**: Supports queries from paper → entity → community → global level
- **Meta-Analysis Integration**: Automatic effect size extraction, forest plots, bias risk assessment, etc.
- **Multi-API Support**: Compatible with any OpenAI-protocol LLM APIs (DeepSeek, Qwen, OpenAI, etc.)

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure API Keys

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

Then edit the `.env` file with your API Key and endpoint.

Or set environment variables directly:

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
export LLM_MODEL=deepseek-v3
```

#### 3. Prepare Data

Place your PDF/Markdown papers into `process/dataset/rTMS/markdown/` directory.

#### 4. Run

```bash
python test.py
```

### Project Structure

```
├── nano_graphrag/                   # Core package
│   ├── keyword_based_graphrag.py    # Keyword-enhanced GraphRAG (core)
│   ├── meta_analysis_graphrag.py    # Meta-analysis GraphRAG
│   ├── graphrag.py                  # Base GraphRAG implementation
│   ├── base.py                      # Base class definitions
│   ├── prompt.py                    # Prompt templates
│   ├── _llm.py                      # LLM interfaces
│   ├── _op.py                       # Operations layer
│   ├── _utils.py                    # Utility functions
│   ├── _storage/                    # Storage backends
│   └── entity_extraction/           # Entity extraction module
├── test.py                          # Main entry script
├── meta_analysis.py                 # Meta-analysis logic
├── outcome_aliases_config.py        # Outcome alias configuration
├── .env.example                     # Environment variable template
└── requirements.txt                 # Dependencies
```

---

## 中文

### 核心功能

- **关键词增强的GraphRAG**：在传统图RAG基础上加入关键词索引和语义索引，实现更精准的跨论文知识检索
- **多层次知识图谱**：支持从论文→实体→社区→全局的多层次查询
- **Meta分析集成**：自动提取效应量、森林图、偏倚风险评估等
- **多API支持**：兼容 OpenAI 协议的各种 LLM API（DeepSeek、通义千问、OpenAI等）

### 快速开始

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置 API 密钥

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

#### 3. 准备数据

将你的 PDF/Markdown 论文放入 `process/dataset/rTMS/markdown/` 目录。

#### 4. 运行

```bash
python test.py
```

### 项目结构

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

---

## License

MIT License

- The GraphRAG core is adapted from [nano-graphrag](https://github.com/gusye1234/nano-graphrag) (MIT License, Copyright (c) 2024 Gustavo Ye)
- Keyword enhancement, meta-analysis integration, and other features are independently developed