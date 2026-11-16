# 第八步：端到端回答质量评估

这一阶段不增加新的问答功能，而是用第四步的 42 道题检验完整链路：

```text
问题 → 混合检索 → Reranker → DeepSeek 生成 → 引用 → 质量评审
```

## 评测维度

- 可回答性：有答案题是否回答、无答案题是否拒答。
- 检索召回：人工标注的正确证据是否进入 Top 5。
- 引用：最终引用是否包含人工标注的制度与关键原文。
- 正确性与忠实度：独立 DeepSeek 评审请求检查答案是否正确、是否超出证据。
- 工程指标：响应时间、输入输出 token、API 或格式错误。

## 运行

```powershell
.\.venv\Scripts\python.exe .\step8_end_to_end_evaluation\evaluate_end_to_end.py
```

调试时可以先运行前几题：

```powershell
.\.venv\Scripts\python.exe .\step8_end_to_end_evaluation\evaluate_end_to_end.py --limit 3
```

## 重要限制

评审模型与生成模型相同，因此自动评审可能有偏差。程序同时使用人工标注的制度标题和关键原文进行确定性检查，但仍不能替代上线前的人工抽查。

## 本次结果

2026-10-14 完成全部 42 题：

- 42 题全部成功调用，无 API 或格式错误。
- 38 道有答案题均回答，正确证据引用率 100%。
- 4 道无答案题全部拒答。
- 自动评审答案正确率、忠实度和引用正确率均为 100%。
- 平均每题 2.577 秒，P95 为 2.951 秒，总耗时 108.254 秒。
- 生成与评审共调用 API 84 次。

结果详见 `evaluation_report.md` 和 `evaluation_report.json`；另对 10 道代表性问题进行了人工抽样，记录在 `manual_spot_check.md`。
