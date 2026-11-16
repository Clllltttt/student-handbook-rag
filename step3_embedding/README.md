# RAG 第三步：Embedding 向量化与本地检索

## 目的

把文本知识块转换成稠密向量，使语义相近的问题和条款能够通过数学计算互相找到。

## 模型

使用 `BAAI/bge-small-zh-v1.5`：

- 中文专用 Embedding 模型。
- 输出 512 维向量。
- 模型权重约 96 MB，适合作为可重复的本地基线。
- 查询使用官方建议的中文检索指令；文档不加指令。
- 使用 `[CLS]` 向量并做 L2 归一化。

本项目当前安装的是 CPU 版 PyTorch，因此实际使用 CPU。254-407 个知识块规模很小，足以完成学习和基线评测。后续可以安装 CUDA 版 PyTorch或升级至 BGE-M3 做对照实验。

## 模型窗口适配

第二步的 254 个语义块按字符数切分，其中 148 个最初超过模型的 512-token 上限。系统没有静默截断，而是停止建库并报告问题。

随后增加了模型适配切分：

- 输入 254 个语义块。
- 152 个父块需要按句子继续拆分。
- 输出 407 个 Embedding 知识块。
- 最大 499 token，全部低于 512-token 硬限制。
- 使用 `parent_chunk_id` 保留与第二步语义块的关系。

## 本地索引

- `index/vectors.npy`：407 × 512 的归一化向量矩阵。
- `index/metadata.jsonl`：每个向量对应的正文、标题和页码。
- `index/index_config.json`：模型及索引参数。

因为向量已归一化，所以余弦相似度可以直接计算：

```text
scores = document_vectors @ query_vector
```

当前数据只有 407 条，直接使用 NumPy 暴力计算能够看清原理且速度足够，无须提前引入 FAISS 或向量数据库。

## 检索评测

使用 6 道人工编写的问题测试 Recall@5，6 道题的正确条款都进入前 5，且全部排在第 1 名。该结果只能证明最小链路正确，不能代表真实系统已达到生产质量；后续需要扩充测试集。

## 运行检索

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe .\step3_embedding\search.py `
  "研究生最长可以读几年？" `
  --index-dir .\step3_embedding\index `
  --model-dir .\models `
  --top-k 5
```

## 文件

- `embedding_model.py`：模型加载、CLS Pooling 和归一化。
- `prepare_embedding_chunks.py`：模型 token 窗口适配。
- `build_index.py`：批量向量化并保存索引。
- `search.py`：查询向量化和余弦相似度排序。
- `evaluate_retrieval.py`：最小检索评测。
- `embedding_chunks.jsonl`：适配后的 407 个知识块。
- `embedding_chunk_report.json`：token 适配统计。
- `evaluation_results.json`：评测明细。

此时仍未调用生成式大模型。系统只能“找到原文”，还不能组织自然语言答案。
