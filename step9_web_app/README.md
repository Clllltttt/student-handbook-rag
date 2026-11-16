# 第九步：本地网页界面与应用封装

英文名称：Local Web Interface & Application Packaging

这一层把第七步已经验证的完整 RAG 管线包装成浏览器应用。浏览器只发送问题、展示回答和来源；Embedding、BM25、Reranker、DeepSeek API Key 和 PDF 都留在本机 Python 后端。

## 启动

最简单的方式：双击 `启动学生手册问答.bat`。

也可以在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\step9_web_app\start_web_app.ps1
```

浏览器地址：`http://127.0.0.1:7860`

按终端中的 `Ctrl+C` 停止服务。

第一次提问会加载约 3.3 GB 的本地模型缓存，当前 CPU 环境实测冷启动约 62 秒；模型常驻后，下一次同类问答实测约 2 秒。启动窗口不要关闭，否则网页后端也会停止。

## 请求链路

1. 浏览器将问题 POST 到 `/api/ask`。
2. Flask 后端第一次收到问题时加载本地 Embedding 和 Reranker 模型。
3. 后端执行向量检索、BM25、RRF 融合和 Reranker。
4. 只把问题与 Top 5 证据发送给 DeepSeek。
5. 后端校验模型 JSON 和引用编号，再把安全的结构化结果返回网页。
6. 网页展示回答、制度名称、手册纸面页码和 PDF 原页入口。

## 安全边界

- 服务只监听 `127.0.0.1`，局域网和公网默认无法访问。
- API Key 仍由项目根目录 `.env` 提供，不会进入 HTML 或 JavaScript。
- 页面显示的是手册纸面页码；打开 PDF 时自动加上 6 页封面与目录偏移。
- 这是开发服务器封装，适合本机学习和演示，不应直接公开到互联网。

## 验证记录

- Python 语法、JavaScript 语法、首页、状态接口和空问题校验通过。
- 桌面 1360 × 900 与移动端 390 × 844 均无横向溢出，浏览器控制台无错误。
- 网页端问题“研究生最长可以读几年？”端到端通过，返回《东华大学研究生学籍管理规定》手册第 98–99 页的一条引用。
- 引用链接正确换算至物理 PDF 第 104 页。

## 文件

- `app.py`：本地 Flask 路由、参数校验和错误处理。
- `rag_service.py`：复用完整 RAG 管线，并让模型只加载一次。
- `templates/index.html`：页面语义结构。
- `static/styles.css`：响应式视觉设计。
- `static/app.js`：提问、状态、答案和引用交互。
- `start_web_app.ps1`：一键启动脚本。
