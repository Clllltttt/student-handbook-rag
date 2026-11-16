# 第七步：基于证据生成答案

完整链路：

```text
问题 → 向量检索 + BM25 → RRF 融合 → Reranker → DeepSeek → 答案与来源
```

默认生成模型为 `deepseek-flash`，接口地址为 `https://api.deepseek.com`。二者都可以通过环境变量修改。

程序显式关闭模型的思考模式。该任务是依据给定证据输出短 JSON，不需要长链路推理；关闭后可以避免内部推理占满 `max_tokens` 导致最终内容为空，同时降低延迟与 token 消耗。

## 1. 配置 API Key

将 `step7_generation/.env.example` 复制为项目根目录的 `.env`：

```powershell
Copy-Item .\step7_generation\.env.example .\.env
```

然后用文本编辑器在 `.env` 中填写真实的 `DEEPSEEK_API_KEY`。不要把 Key 发到聊天或写进 Python 文件。

## 2. 检查配置

```powershell
.\.venv\Scripts\python.exe .\step7_generation\validate_setup.py
```

检查程序只显示是否已配置，不会打印 Key。

## 3. 不调用 API 的演练

```powershell
.\.venv\Scripts\python.exe .\step7_generation\answer_question.py `
  "宿舍每年的住宿费是多少？" --dry-run
```

这会完成本地检索和重排序，并显示将发送给 DeepSeek 的证据与提示词，不产生 API 费用。

## 4. 正式提问

```powershell
.\.venv\Scripts\python.exe .\step7_generation\answer_question.py `
  "高校招生工作应遵循哪些基本原则？"
```

程序要求模型输出结构化 JSON，再由本地代码验证引用编号，并最终显示答案、制度名称和手册页码。资料不足时要求模型明确拒答。

## 安全边界

- API Key 只从环境变量或根目录 `.env` 读取。
- `.env` 已加入根目录 `.gitignore`，不会被正常 Git 提交。
- 发送给 DeepSeek 的内容包括用户问题和 Top 5 手册证据，不会发送整本 PDF。
- 引用存在只能证明模型引用了某段资料；答案是否完全忠于原文仍需在第八步进行端到端评估。

## 在线验证结果

2026-10-14 已完成两道在线冒烟测试：

- 有答案问题“高校招生工作应遵循哪些基本原则？”正确回答并引用第 49 页。
- 无答案问题“宿舍每年的住宿费是多少？”明确指出手册未提供金额，没有编造数字。

详细结果见 `online_validation_report.md`，结构化原始结果保存在 `smoke_answerable.json` 和 `smoke_unanswerable.json`。
