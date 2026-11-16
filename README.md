# 学生手册智能问答系统（RAG）

面向学生查阅校内规章制度的本地问答应用。围绕奖助政策、学籍管理和校园生活等查询场景，将长篇学生手册转化为可检索知识，提供自然语言回答、原文引用和 PDF 原页跳转；资料不足时明确说明未找到支持信息。

本项目由个人独立开发，是用于学习、功能验证和产品演示的原型，目前没有公开在线服务。当前实现针对《东华大学研究生手册（2023年版）》，不是学校官方服务。

## 用户问题与产品设计

| 用户问题 | 产品设计 | 使用价值 |
| --- | --- | --- |
| 手册制度多，难以快速找到具体条款 | 自然语言提问，结合语义检索和关键词检索 | 降低查阅长文档的成本 |
| 生成式回答可能出错，难以判断依据 | 展示引用制度、原文和页码，支持打开 PDF 原页 | 方便核验答案真实性 |
| 手册未覆盖的问题也可能被模型回答 | 限制模型依据检索证据回答，证据不足时拒答 | 减少编造具体规定的风险 |
| 网页每次请求重新加载模型会增加等待 | 后端复用已加载模型，前端展示请求状态 | 改善重复查询体验 |

引用编号与 JSON 格式会在后端校验；这些校验不能证明答案一定正确，用户仍应核对原文。拒答依赖模型对证据的判断，也存在误判可能。

## 查询案例

- **奖学金相关制度**：用于检验能否找到奖助政策条款，并给出可追溯的回答依据。
- **“研究生最长可以读几年？”**：已有本地网页验证记录，展示对应制度与原页入口。
- **“宿舍每年的住宿费是多少？”**：既有测试记录显示，系统说明手册没有提供具体金额，不编造收费数字。

以上用于说明已验证的行为，生成结果可能因模型版本和输入措辞而变化。项目保留各阶段评测脚本；当前公开版本不附带原始手册、派生知识库和历史评测数据，不宣称生产环境准确率。

## 工作流程

```text
学生手册 PDF → 按制度提取 → 结构化切分 → Embedding 窗口适配
                                              ↓
自然语言问题 → 向量检索 + BM25 → RRF 融合 → 重排序
                                              ↓
                                问题 + 候选证据 → 大模型生成
                                              ↓
                                JSON / 引用编号校验
                                              ↓
                          回答与原文引用 / 证据不足提示
                                              ↓
                                    PDF 原页核验
```

Embedding 使用 `BAAI/bge-small-zh-v1.5`，重排序使用 `BAAI/bge-reranker-base`；向量以 NumPy 矩阵保存。生成层通过 OpenAI 兼容 SDK 调用 DeepSeek，网页使用 Flask 与原生 HTML/CSS/JavaScript。

## 本地运行

以下命令在项目根目录执行，按 Windows PowerShell 编写。

### 1. 安装环境

建议使用 Python 3.12。解析 PDF 另外需要 Poppler 的 `pdftotext` 命令，并确保它在 PATH 中可用。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
pdftotext -v
```

依赖版本来自现有项目配置，本文未在全新环境重新安装验证。模型需要从 Hugging Face 下载；请预留数 GB 磁盘空间，首次下载与加载需要等待。

### 2. 准备资料并建立知识库

公开仓库不分发第三方手册、提取全文、模型权重与索引。按 [资料说明](data/README.md) 将有权使用的对应版本手册放到 `data/2023研究生手册.pdf`。当前解析规则依赖该版本目录和页码，不能直接用于任意 PDF。

```powershell
.\.venv\Scripts\python.exe step1_document_ingestion/extract_full_handbook.py data/2023研究生手册.pdf step1_document_ingestion
.\.venv\Scripts\python.exe step2_chunking/chunk_handbook.py step1_document_ingestion/handbook_documents.jsonl step2_chunking
.\.venv\Scripts\python.exe step3_embedding/prepare_embedding_chunks.py step2_chunking/handbook_chunks.jsonl step3_embedding/embedding_chunks.jsonl models
.\.venv\Scripts\python.exe step3_embedding/build_index.py step3_embedding/embedding_chunks.jsonl step3_embedding/index models
.\.venv\Scripts\python.exe step5_hybrid_retrieval/build_bm25_index.py --metadata step3_embedding/index/metadata.jsonl --output step5_hybrid_retrieval/index/bm25_index.json
```

问答阶段默认只读取本地重排序模型，首次运行前还需下载其缓存：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-base', cache_dir='models')"
```

### 3. 配置生成模型

```powershell
Copy-Item .env.example .env
```

在本机 `.env` 填写 `DEEPSEEK_API_KEY`，并按账号实际可用模型设置 `DEEPSEEK_MODEL`。示例中的模型名不保证对所有账号可用。真实密钥不应写进代码或提交到 Git。

生成步骤会向所配置的服务发送问题与检索证据，并产生 API 调用费用；未配置密钥时可先只验证本地链路：

```powershell
.\.venv\Scripts\python.exe step7_generation/answer_question.py "研究生最长可以读几年？" --dry-run
```

### 4. 启动网页

```powershell
.\.venv\Scripts\python.exe step9_web_app/app.py
```

打开 `http://127.0.0.1:7860`，按 `Ctrl+C` 停止。也可使用 `step9_web_app/启动学生手册问答.bat`。首次提问会加载模型，后续复用模型实例；实际耗时取决于设备和 API。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `step1_document_ingestion` | PDF 制度提取与页码保留 |
| `step2_chunking` | 按条款结构切分知识块 |
| `step3_embedding` | 模型窗口适配、向量建库与检索 |
| `step4_evaluation` | 检索测试集校验及评测脚本 |
| `step5_hybrid_retrieval` | BM25、RRF 融合和对比评测 |
| `step6_reranking` | 交叉编码器重排序 |
| `step7_generation` | 基于证据生成、引用校验及配置示例 |
| `step8_end_to_end_evaluation` | 端到端评测与模型评审脚本 |
| `step9_web_app` | 本地问答网页与服务封装 |

各阶段 README 记录了原始开发过程。其历史结果依赖本地数据与当时环境；公开仓库中的评测脚本需要自行准备相应数据后运行。

## 当前边界与后续方向

- 当前是本地单用户演示，Flask 仅监听 `127.0.0.1`，没有用户登录、访问配额与生产部署配置。
- 解析范围、展示名称、制度分段和 PDF 页码偏移针对特定手册；泛化到其他资料需要调整。
- 尚缺真实用户验收、非规范提问测试、知识库更新与过期制度处理。
- 可继续验证多轮追问、检索失败提示及冷启动体验，已有能力不代表这些方向已经完成。

## 资料与许可

第三方手册不随仓库发布，模型请从原发布渠道获取并遵循各自许可。当前未为项目代码指定开源许可证；公开代码便于浏览学习，后续再确定允许他人使用、修改和分发的授权条款。
