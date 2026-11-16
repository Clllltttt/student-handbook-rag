# 第五步：混合检索

这一阶段在第三步的向量检索之外增加 BM25 关键词检索，并使用 Reciprocal Rank Fusion（RRF）融合两份排名。

## 为什么需要两条检索通道

- 向量检索理解近义表达和语义改写，但可能忽略法规名称、数字和精确措辞。
- BM25 对问题和原文共有的关键词敏感，但不擅长理解没有字面重合的表达。
- RRF 使用排名而不是原始分数，适合融合两种分数量纲不同的检索器。

中文分词采用可直接阅读的字符二元组、三元组方案，不依赖第三方分词库。制度标题重复 3 次，给予标题词适度权重。

## 本次结果

测试集仍为第四步的 42 题，其中 38 道有答案、4 道无答案诊断题：

| 模式 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| 纯向量 | 71.1% | 86.8% | 94.7% | 97.4% | 0.793 |
| 纯 BM25 | 84.2% | 97.4% | 97.4% | 100.0% | 0.899 |
| RRF 混合 | 81.6% | 94.7% | 100.0% | 100.0% | 0.885 |

原来的两个失败问题中，“高校招生原则”的正确块从向量第 24 名提升到混合第 4 名；“网络数据的性质”从第 6 名提升到第 3 名。混合检索已经保证 38 道正样本的正确证据全部进入 Top 5，下一步用 Reranker 改善 Top 1 排序。

## 构建 BM25 索引

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe .\step5_hybrid_retrieval\build_bm25_index.py `
  --metadata .\step3_embedding\index\metadata.jsonl `
  --output .\step5_hybrid_retrieval\index\bm25_index.json
```

## 运行混合检索

```powershell
.\.venv\Scripts\python.exe .\step5_hybrid_retrieval\hybrid_search.py `
  "高校招生工作应遵循哪些基本原则？" `
  --dense-index-dir .\step3_embedding\index `
  --model-dir .\models `
  --bm25-index .\step5_hybrid_retrieval\index\bm25_index.json `
  --metadata .\step3_embedding\index\metadata.jsonl
```

## 重新评估

```powershell
.\.venv\Scripts\python.exe .\step5_hybrid_retrieval\evaluate_hybrid.py `
  --cases .\step4_evaluation\test_cases.jsonl `
  --dense-index-dir .\step3_embedding\index `
  --model-dir .\models `
  --bm25-index .\step5_hybrid_retrieval\index\bm25_index.json `
  --metadata .\step3_embedding\index\metadata.jsonl `
  --output-dir .\step5_hybrid_retrieval
```
