# RAG 第二步：知识切分（Chunking）

## 目的

把第一步生成的 30 份长制度拆成适合检索的知识块。切分过大会混入无关内容，切分过小会丢失上下文。

## 本项目的切分策略

采用“结构优先，长度兜底”：

1. 先识别章、节、条、款和列表边界。
2. 修复被 PDF 分页截断的句子。
3. 合并相邻的短条款，保留足够上下文。
4. 目标约 600 字，偏好不少于 180 字，绝不超过 900 字。
5. 只有单个超长句无法自然拆分时，才使用 80 字重叠。
6. 每个知识块继承制度标题、类别和纸面页码。

## 结果

- 输入：30 份制度。
- 输出：254 个知识块。
- 最短：180 字。
- 中位数：600 字。
- 平均：约 524 字。
- 最长：887 字。
- 空知识块：0。
- 超过 900 字：0。
- 重复知识块 ID：0。

## 文件

- `chunk_handbook.py`：切分程序。
- `handbook_chunks.jsonl`：切分结果，一行一个知识块。
- `chunking_stats.json`：切分统计。
- `inspect_chunks.py`：按关键词人工检查知识块。

## 人工检查示例

```powershell
python .\inspect_chunks.py .\handbook_chunks.jsonl "住宿费"
python .\inspect_chunks.py .\handbook_chunks.jsonl "最长学习年限"
python .\inspect_chunks.py .\handbook_chunks.jsonl "国家奖学金"
```

此时仍然没有向量数据库和大模型。我们先确认知识块是否语义完整，再进入第三步“Embedding 向量化”。
