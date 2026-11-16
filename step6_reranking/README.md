# 第六步：Reranker 重排序

混合检索负责从全部知识块中快速召回少量候选；Reranker 将问题与每个候选原文一起输入 Cross-Encoder，直接判断两者相关性，再重新排序。本次评测使用 Top 5，因为第五步的测试集 Recall@5 已达到 100%。

模型：`BAAI/bge-reranker-base`。选择 base 版是为了在当前 CPU 环境中兼顾中文效果与运行时间。

## 本次结果

| 模式 | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---:|---:|---:|---:|
| 混合检索 | 81.6% | 94.7% | 100.0% | 0.885 |
| Reranker | 89.5% | 100.0% | 100.0% | 0.939 |

42 题在 CPU 上耗时 117.4 秒，平均 2.80 秒/题。招生原则问题从第 4 名提升到第 1 名。仍有 4 题没有排到第一，说明通用 Reranker 会改善总体排序，但不会保证每道题都改善。

## 运行查询

```powershell
.\.venv\Scripts\python.exe .\step6_reranking\rerank_search.py `
  "高校招生工作应遵循哪些基本原则？" `
  --dense-index-dir .\step3_embedding\index `
  --embedding-model-dir .\models `
  --bm25-index .\step5_hybrid_retrieval\index\bm25_index.json `
  --metadata .\step3_embedding\index\metadata.jsonl `
  --reranker-model-dir .\models
```

## 设计边界

Reranker 只能重新排列已经召回的候选，无法找回不在候选集合中的正确证据。因此第五步先提高召回率，第六步再提高第一名的准确率。
