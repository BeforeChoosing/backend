# 选择之前 API

后端负责本机 Demo 的完整数据与大模型链路：

```text
经历输入 → 能力卡确认 → 本地岗位 RAG → 固定任务库选题 → 五步试路 → Qwen 评价 → 成长复盘 → Observed Evidence
```

后端使用 Conda 管理 Python 环境，当前提供 FastAPI API、Qwen 网关、四个独立 Agent 和本地岗位知识库检索。

四个 Agent 的职责和写入边界如下：

| Agent | 职责 | 边界 |
|---|---|---|
| `ProfileAgent` | 从用户经历中整理候选能力卡 | 不确认卡片，不写入长期画像 |
| `CareerAgent` | 结合已确认能力卡和岗位 RAG 解释下一方向 | 不选择或改写任务，不补写岗位事实 |
| `TrialAgent` | 按固定 Rubric 和 L1–L5 锚点评价单次任务 | 不形成长期能力等级或岗位认证 |
| `ReflectionAgent` | 将任务评价整理成证据变更提案 | 不修改评价分数，不直接更新已确认能力卡 |

四个 Agent 可以共用同一个百炼 Qwen 模型，但各自拥有独立执行类、提示词版本、输入输出契约和校验规则。任务提交时先由 `TrialAgent` 评价，再由 `ReflectionAgent` 生成新增、加强、冲突或仍待验证的证据提案。复盘调用失败时，后端保留已经完成的任务评价，并写入明确标注为 `deterministic_fallback` 的保守复盘结果。

当前版本仅用于 macOS、Linux 或 Windows 电脑上的本机运行，不涉及服务器部署。后端默认监听 `127.0.0.1:8000`。

## 前置条件

- macOS/Linux 或 Windows 已安装 Conda（Miniforge 或 Anaconda）。
- 已准备阿里云百炼/DashScope API Key。

## 第一次安装

在当前目录执行：

```bash
conda env create -f environment.yml
conda activate before-choosing-demo
cp .env.example .env
```

Windows PowerShell 使用以下复制命令：

```powershell
conda env create -f environment.yml
conda activate before-choosing-demo
Copy-Item .env.example .env
```

然后编辑 `.env`，只在本机填写密钥：

```env
DASHSCOPE_API_KEY=你的百炼密钥
QWEN_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
LLM_REQUEST_TIMEOUT=45
PROFILE_DB_PATH=profile.db
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

`.env` 不会提交到 Git。

`.env` 是后端本机配置文件。百炼密钥只保存在该文件中，由后端进程读取；不要复制到前端 `.env.local`。

本地岗位知识库默认位于仓库内的 `knowledge/public/`，索引文件为 `knowledge.db`。两者均不包含密钥，索引文件已加入 Git 忽略规则。

如果环境已经存在，后续只需：

```bash
conda activate before-choosing-demo
```

也可以不激活环境，直接使用 `conda run`：

```bash
conda run -n before-choosing-demo python -m uvicorn app.main:app --reload --port 8000
```

## 启动后端

激活环境后执行：

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

后端地址：<http://127.0.0.1:8000>

接口文档：<http://127.0.0.1:8000/docs>

停止服务：在运行窗口按 `Ctrl+C`。

## 检查服务

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Windows PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

正常返回示例：

```json
{
  "status": "ok",
  "service": "选择之前 API",
  "qwen_configured": true,
  "model": "qwen-plus"
}
```

## 调用候选卡接口

```bash
curl -X POST http://127.0.0.1:8000/api/v1/profile/proposals \
  -H 'Content-Type: application/json' \
  -d '{
    "experience_text": "我在校园项目中访谈用户并根据反馈调整了方案，最后完成了可用原型。",
    "target_role": "AI Native 产品经理"
  }'
```

接口只返回候选证据卡，不会直接写入已确认画像。缺少 `DASHSCOPE_API_KEY`、网络不可用或 Qwen 输出无法通过结构化校验时，会返回明确错误，不生成伪造结果。

## 上传材料提取

`POST /api/v1/profile/materials/extract` 接收最大 20MB 的 PDF、Word (`.docx`)、Markdown 或 TXT 文档，只在内存中提取可复制文本，并把最多 12000 字返回前端供用户核对。扫描件 OCR、旧版 `.doc` 和外部链接抓取当前不在支持范围；接口不会把原文件或提取结果直接写入长期画像。

## 12 个固定试路任务与动态选题

当前 Demo 接入 CoachAgent 任务库中的 12 个已校准任务，覆盖 Feature、Application / Agent、Platform / Developer、Model / Eval / Data 四类 AI 产品经理方向。任务材料、五步作答 Schema、中途事件、三级 Coach 提示、Rubric 权重和 L1–L5 行为锚点均来自 Demo 资料；模拟业务数据和案例在接口中明确标识。

进入 03 模块后分为两个阶段：第一阶段从已确认能力卡中选择 1–4 张，记录准备如何使用这些能力和本次待验证假设；第二阶段进入现有五步真实任务工作台。能力出牌是任务前预期，不直接计入分数或能力等级。`TrialAgent` 只评价真实任务中的可观察行为，`ReflectionAgent` 再对比预期与实际证据。

后端选择器根据已确认能力卡、待验证描述、目标岗位、最近评价中的主测能力/等级/置信度/下一步建议和已完成任务进行确定性排序。Qwen 不参与任务选择，不生成或改写题目。同样输入得到同样排序；存在未完成任务时会跳过已形成 Observed Evidence 的任务。

试路接口：

- `GET /api/v1/trial/catalog`：读取 12 个固定任务。
- `GET /api/v1/trial/catalog/{task_id}`：读取单个任务的材料与作答结构。
- `POST /api/v1/trial/recommendations`：使用已确认能力卡选择下一任务。
- `POST /api/v1/trial/workbench/sessions`：创建本机作答会话。
- `GET /api/v1/trial/workbench/sessions/{session_id}`：恢复会话。
- `PUT /api/v1/trial/workbench/sessions/{session_id}/answer`：保存五步作答、材料查看/引用和修改次数。
- `POST /api/v1/trial/workbench/sessions/{session_id}/coach`：使用并记录一级、二级或三级提示。
- `POST /api/v1/trial/workbench/sessions/{session_id}/event`：触发中途事件。
- `POST /api/v1/trial/workbench/sessions/{session_id}/submit`：由后端依次调用 `TrialAgent` 和 `ReflectionAgent`，写回任务评价、复盘提案与 `Observed Evidence`。

Qwen 只评价固定任务中的可观察行为。接口返回各 Rubric 的分项任务分、主测能力的 `Observed Level`、证据依据、Coach 依赖和置信度，不计算单题总分，不把一次任务直接等同为 `Current Level`，也不输出岗位匹配百分比或企业认证结论。后端会丢弃模型自创的维度，并用任务库中的权重、主测能力和 L1–L5 锚点覆盖模型输出。复盘提案中的能力名和证据引用同样经过白名单校验，不能引用模型自创的来源。

原 `A-02` 专用接口仍保留用于兼容已有本机会话，新主流程统一使用 `/trial/workbench/` 接口。

## 本地岗位 RAG 与职业推演

职业探索页只读取已确认能力卡。后端将能力卡内容与本地岗位资料组合成检索词，在 SQLite FTS5 索引中检索 AI 产品经理岗位片段，再把检索片段和引用 ID 交给 Qwen 生成结构化推演。前端不会直接请求百炼，也不会接触 API Key。

知识库资料来自项目提供的 `公共RAG知识库` 解压内容，已按文档登记 `document_id`、资料级别和来源说明。当前岗位文档属于公开资料交叉归纳稿，原始 JD 链接尚待补齐，因此界面会显示资料级别和来源提示，不将归纳稿当作官方岗位结论。

首次安装或更新 Markdown 资料后，在后端仓库根目录执行索引构建：

```bash
conda run -n before-choosing-demo python -m app.knowledge.indexer
```

Windows PowerShell 使用同一条命令。服务启动时也会检查文件指纹并自动建立缺失索引。

职业推演接口：

- `POST /api/v1/career/recommendations`：提交 1–4 个已确认能力卡 ID，返回 AI 产品经理路径摘要、支持性判断、未知项、动态选择的下一任务和本地引用片段。

## 本地画像持久化

用户点击“加入能力库”后，已确认卡片会写入本机 SQLite 文件。`PROFILE_DB_PATH` 用于指定文件位置，默认值为 `profile.db`；该文件已加入 Git 忽略规则，不会提交到仓库。

画像接口：

- `GET /api/v1/profile/cards`：读取已确认卡片，页面刷新后恢复。
- `GET /api/v1/profile/overview`：读取已确认卡片、任务评价证据、评价结果和已完成任务 ID，供个人档案和下一任务选择使用。
- `POST /api/v1/profile/cards/confirm`：确认并保存候选卡片，同时记录画像版本。
- `PATCH /api/v1/profile/cards/{card_id}`：更新卡片文字内容。
- `DELETE /api/v1/profile/cards/{card_id}`：删除卡片。

重置本机 Demo 画像时，先停止后端，再删除 `profile.db`；Windows PowerShell 使用 `Remove-Item profile.db`。

## 运行测试

```bash
conda run -n before-choosing-demo pytest -q
```

## 常见问题

| 现象 | 处理方式 |
|---|---|
| `未配置 DASHSCOPE_API_KEY` | 检查当前目录的 `.env`，或确认启动时使用了 `before-choosing-demo` 环境 |
| 前端提示无法连接后端 | 确认后端正在 `8000` 端口运行 |
| 前端收到 `503` | 后端可访问，但百炼密钥、额度或网络配置有问题 |
| 端口被占用 | 将启动命令中的 `--port 8000` 换成其他端口，并同步修改前端 `VITE_API_BASE_URL` |
| 职业推演提示知识库未准备完成 | 在后端仓库根目录运行 `conda run -n before-choosing-demo python -m app.knowledge.indexer`，确认 `knowledge/public/` 中存在 Markdown 资料 |
