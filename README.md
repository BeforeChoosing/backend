# 选择之前 API

后端负责本机 Demo 的完整数据与大模型链路：

```text
经历输入 → 能力卡确认 → 本地岗位 RAG → 固定任务库选题 → 三轮能力应用推演 → 五步试路 → Qwen 评价 → 成长复盘 → Observed Evidence
```

后端使用 Conda 管理 Python 环境，当前提供 FastAPI API、Qwen 网关、四个独立 Agent、本地岗位知识库检索、TrialAgent 评测与多模态证据定位。
岗位 RAG 采用本地 Markdown、SQLite FTS5、SQLite 向量表和百炼检索模型组合：资料与向量索引留在本机，百炼只接收必要的文本请求，不使用托管知识库或本地下载模型。

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
- 扫描 PDF 的页面渲染依赖 `PyMuPDF`，已列入 `environment.yml`；更新已有环境后执行 `conda env update -f environment.yml --prune`。

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
TRIAL_BASE_MODEL=qwen-plus
TRIAL_SFT_MODEL=
TRIAL_VERIFIER_MODEL=
TRIAL_TEACHER_MODEL=qwen3-vl-plus
TRIAL_REVIEW_MODEL=qwen3-vl-235b-a22b-instruct
TRIAL_TEACHER_PROMPT_VERSION=trial-teacher-v1
TRIAL_TEACHER_CACHE_PATH=datasets/trial_agent/v1/teacher_cache.sqlite3
TRIAL_VERIFIER_MIN_EVIDENCE_COVERAGE=0.75
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
BAILIAN_EMBEDDING_URL=
BAILIAN_EMBEDDING_MODEL=qwen3.7-text-embedding
BAILIAN_EMBEDDING_DIMENSION=1024
BAILIAN_EMBEDDING_BATCH_SIZE=20
BAILIAN_RERANK_URL=
BAILIAN_RERANK_MODEL=qwen3-rerank
BAILIAN_VISION_MODEL=qwen-vl-ocr
MULTIMODAL_MAX_PAGES=8
RAG_RETRIEVER_MODE=vector
RAG_CANDIDATE_LIMIT=20
RAG_RERANK_LIMIT=5
LLM_REQUEST_TIMEOUT=45
PROFILE_DB_PATH=profile.db
KNOWLEDGE_DIR=knowledge/public
KNOWLEDGE_DB_PATH=knowledge.db
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

`BAILIAN_EMBEDDING_URL` 和 `BAILIAN_RERANK_URL` 留空时，会自动沿用 `DASHSCOPE_BASE_URL` 的工作空间 Host；使用百炼专属工作空间时只需填写聊天地址和密钥。

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

## 一键验收

先启动前端和后端，再在后端仓库根目录执行验收脚本。默认检查环境变量、百炼地址、本地 RAG、12 个固定任务、四个 Agent，以及前后端连通性；默认不会调用任何模型，不产生模型费用。

macOS/Linux：

```bash
conda activate before-choosing-demo
python scripts/check_demo.py
```

Windows PowerShell：

```powershell
conda activate before-choosing-demo
python .\scripts\check_demo.py
```

前后端尚未启动时，可以只检查静态配置和本地数据：

```bash
python scripts/check_demo.py --skip-services
```

Windows PowerShell：

```powershell
python .\scripts\check_demo.py --skip-services
```

需要确认百炼真实连通性时，显式增加 `--live-qwen`。该参数只执行 1 次有意义的 Qwen JSON 调用，会产生少量费用：

```bash
python scripts/check_demo.py --live-qwen
```

Windows PowerShell：

```powershell
python .\scripts\check_demo.py --live-qwen
```

需要确认本地向量检索和重排链路时，显式增加 `--live-rag`。该参数只执行 1 次查询向量和 1 次重排调用，并要求已经建立本地向量索引：

```bash
python scripts/check_demo.py --live-rag
```

Windows PowerShell：

```powershell
python .\scripts\check_demo.py --live-rag
```

## 01 能力探索与候选卡

`POST /api/v1/profile/exploration/messages` 使用当前经历草稿和最多 12 条补充对话生成一条聚焦引导。接口由 `ProfileAgent` 负责自然表达和候选事实整理；是否继续追问、是否达到候选卡整理条件由后端证据覆盖控制器确定，不由模型结果直接决定。

控制器只读取 `experience_text` 和 `role=user` 的历史消息，排除助手文本，按以下七个维度计算覆盖状态：`ownership`（本人承担）、`decision`（判断依据）、`constraint`（限制条件）、`collaboration`（协作过程）、`result`（实际结果）、`transfer`（后续迁移）和 `evidence`（可核对材料）。每个维度返回 `missing`、`weak` 或 `sufficient`；`confirmed` 只在用户确认候选能力卡后成立，不由探索接口写入。服务端要求本人行动、判断依据、结果和可核对证据达到 `sufficient`，并至少出现一类协作或限制信息，才将 `ready_for_proposal` 置为 `true`。响应中的 `coverage` 可直接用于前端展示覆盖进度。

请求可选传入已聚焦过的 `focus_history`，控制器会在仍有其他缺口时跳过这些维度，避免连续重复同一补充方向。旧版前端不传该字段时仍保持兼容，覆盖状态和完成条件照常由服务端重算。模型每轮仍只调用一次，响应缓存键包含覆盖控制版本相关输入，不会因控制器判断而增加固定调用。

完成补充后调用候选卡接口：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/profile/proposals \
  -H 'Content-Type: application/json' \
  -d '{
    "experience_text": "我在校园项目中访谈用户并根据反馈调整了方案，最后完成了可用原型。",
    "target_role": "AI Native 产品经理"
  }'
```

候选卡接口同样会复用完全相同输入的有效模型结果。接口只返回候选证据卡，不会直接写入已确认画像。缺少 `DASHSCOPE_API_KEY`、网络不可用或 Qwen 输出无法通过结构化校验时，会返回明确错误，不生成伪造结果。

## 上传材料提取与多模态证据

`POST /api/v1/profile/materials/extract` 接收最大 20MB 的 PDF、Word (`.docx`)、Markdown 或 TXT 文档，只在内存中提取可复制文本，并把最多 12000 字返回前端供用户核对。接口不会把原文件或提取结果直接写入长期画像。

`POST /api/v1/profile/materials/multimodal-extract` 使用 `.env` 中的 `BAILIAN_VISION_MODEL`（默认 `qwen-vl-ocr`）处理图片材料和扫描 PDF。扫描 PDF 最多渲染 `MULTIMODAL_MAX_PAGES` 页；图片或页面以一次视觉请求发送，结果保留 `page`、归一化 `bbox`、连续文字 `quote`、来源哈希和 `source_ref`。每条结果状态固定为 `candidate`，前端核对前不会进入能力卡或职业推荐。

前端上传普通文字 PDF 时只调用文字提取；检测到 PDF 没有文字层，或上传 PNG/JPG/WebP 时才调用一次 Qwen-VL。这样不会为同一份可复制文本重复支付视觉调用。旧版 `.doc` 和外部链接抓取仍不在支持范围。

多模态定位评测使用独立的人工标注集，金标准记录材料标识、页码、归一化区域和连续文字。离线脚本按同材料、同页、区域 IoU 与文字相似度进行一对一匹配，输出页码命中率、定位 IoU、证据精确率/召回率、材料覆盖率和页面覆盖率；样例只验证报告链路，不作为精度结论：

```bash
conda run -n before-choosing-demo python scripts/evaluate_multimodal.py \
  --cases datasets/multimodal/eval/cases.example.jsonl \
  --predictions datasets/multimodal/eval/predictions.example.jsonl
```

Windows PowerShell：

```powershell
conda run -n before-choosing-demo python .\scripts\evaluate_multimodal.py `
  --cases .\datasets\multimodal\eval\cases.example.jsonl `
  --predictions .\datasets\multimodal\eval\predictions.example.jsonl
```

正式评测需由人工审核材料后生成 `cases.jsonl`，再导入实际 Qwen-VL 预测；报告输出到已忽略的 `evaluation-results/multimodal-v1/`，评测本身不调用百炼。

## 12 个固定试路任务与动态选题

当前 Demo 接入 CoachAgent 任务库中的 12 个已校准任务，覆盖 Feature、Application / Agent、Platform / Developer、Model / Eval / Data 四类 AI 产品经理方向。任务材料、五步作答 Schema、中途事件、三级 Coach 提示、Rubric 权重和 L1–L5 行为锚点均来自 Demo 资料；模拟业务数据和案例在接口中明确标识。

进入 03 模块后分为两个阶段。第一阶段包含三轮能力应用推演，每轮从完整的已确认能力卡库中选择 1–3 张卡牌。后端使用固定任务 Rubric 的前三项评价维度作为答案依据，并根据能力卡类别返回高度适用、部分适用或关联较弱的结果；该过程不调用 Qwen。第二阶段进入现有五步真实任务工作台。能力出牌是任务前判断，不直接计入分数或能力等级。`TrialAgent` 只评价真实任务中的可观察行为，`ReflectionAgent` 再对比预期与实际证据。

后端选择器根据已确认能力卡、待验证描述、目标岗位、最近评价中的主测能力/等级/置信度/下一步建议和已完成任务进行确定性排序。Qwen 不参与任务选择，不生成或改写题目。同样输入得到同样排序；存在未完成任务时会跳过已形成 Observed Evidence 的任务。

试路接口：

- `GET /api/v1/trial/catalog`：读取 12 个固定任务。
- `GET /api/v1/trial/catalog/{task_id}`：读取单个任务的材料与作答结构。
- `POST /api/v1/trial/recommendations`：使用已确认能力卡选择下一任务。
- `POST /api/v1/trial/workbench/sessions`：创建本机作答会话。
- `GET /api/v1/trial/workbench/sessions/{session_id}`：恢复会话。
- `PUT /api/v1/trial/workbench/sessions/{session_id}/answer`：保存三轮能力选择与匹配反馈，以及五步作答、材料查看/引用和修改次数。
- `POST /api/v1/trial/workbench/sessions/{session_id}/coach`：使用并记录一级、二级或三级提示。
- `POST /api/v1/trial/workbench/sessions/{session_id}/event`：触发中途事件。
- `POST /api/v1/trial/workbench/sessions/{session_id}/submit`：由后端依次调用 `TrialAgent` 和 `ReflectionAgent`，写回任务评价、复盘提案与 `Observed Evidence`。

Qwen 只评价固定任务中的可观察行为。接口返回各 Rubric 的分项任务分、主测能力的 `Observed Level`、证据依据、Coach 依赖和置信度，不计算单题总分，不把一次任务直接等同为 `Current Level`，也不输出岗位匹配百分比或企业认证结论。后端会丢弃模型自创的维度，并用任务库中的权重、主测能力和 L1–L5 锚点覆盖模型输出。复盘提案中的能力名和证据引用同样经过白名单校验，不能引用模型自创的来源。

03 评价采用 `trial-evidence-v1` 证据协议。提交时后端先从已确认能力卡、三轮能力应用、五步作答、材料引用和事件响应组装 `TrialEvidenceBundle`，为每条内容生成稳定的证据 ID；`TrialAgent` 只能引用证据目录中的 ID。随后 `TrialScoringService` 校验 Rubric 维度和权重、过滤不存在的引用，并在缺少实际作答证据时限制分数和 Observed Level。评价结果中的 `ability_applications` 会标记每张能力卡为“已应用”“部分应用”或“未形成证据”，同时返回关联挑战、作答和事件证据；`Observed Evidence` 保存完整证据条目供前端回看。演示模式使用同一协议生成固定示例结果，正式模式在同一校验链路中调用 Qwen。

相同能力卡、岗位资料和任务选择结果会复用已经通过结构化校验的职业推荐；相同任务定义、作答和提示词版本会复用已经通过校验的任务评价。缓存保存在本机 `PROFILE_DB_PATH` 指定的 SQLite 文件中。单个会话的并发提交会串行处理，已提交会话直接返回原评价。Coach 提示只在用户主动点击时记录，不进入自动缓存调用。

原 `A-02` 专用接口仍保留用于兼容已有本机会话，新主流程统一使用 `/trial/workbench/` 接口。

## TrialAgent 评测、SFT 与低置信度校验

评测数据与训练数据分开管理。所有案例必须来自固定 12 个任务库；教师模型生成的内容只作为银标候选，经过结构化校验、证据引用校验、重复过滤和人工抽检后，才可以进入 SFT。多模态模型只用于运行时材料提取和证据定位，不进入这条文本评价训练链路。标注边界、输入 JSONL 格式和 SFT 分片命令见 [`datasets/trial_agent/v1/README.md`](datasets/trial_agent/v1/README.md)。本地缓存、标签和生成目录均已忽略，不会提交训练产物。

统一评测报告支持四组对照：`base_qwen`（基线 Qwen）、`prompt_hardened`（当前提示词）、`sft`（百炼部署后的 TrialAgent 微调模型）和 `sft_validator`（SFT 加证据校验链）。报告统计结构合法率、分项分数 MAE、Observed Level 准确率及 ±1 命中率、证据精确率/召回率、无效引用和平均 API 调用次数，并记录数据集哈希、模型、提示词版本和 Git 提交号。

离线汇总不会调用模型：

```bash
conda activate before-choosing-demo
python scripts/evaluate_trial_agent.py \
  --cases datasets/trial_agent/eval/cases.example.jsonl \
  --predictions datasets/trial_agent/eval/predictions.example.jsonl
```

Windows PowerShell：

```powershell
conda activate before-choosing-demo
python .\scripts\evaluate_trial_agent.py `
  --cases .\datasets\trial_agent\eval\cases.example.jsonl `
  --predictions .\datasets\trial_agent\eval\predictions.example.jsonl
```

只有明确使用 `--live` 才会按案例和方案调用 Qwen。四组对照会产生 `4 × 案例数` 次 TrialAgent 请求；未设置 `TRIAL_SFT_MODEL` 时不会启动 SFT 组。报告输出到 `evaluation-results/trial-agent-v1/`，该目录已忽略。

低置信度校验先在本机运行：检查 Rubric 是否完整、权重是否一致、证据引用是否属于服务端目录、分数是否有证据支撑以及证据覆盖率。命中任一条件时，评价会返回 `verification.status=needs_review` 和原因码；只有设置 `TRIAL_VERIFIER_MODEL` 才会再调用一次校验模型，且结果按答案与证据缓存，重复提交不会重复付费。校验模型不重新评分，也不能直接修改能力卡。

### 百炼 SFT 操作边界

`TRIAL_SFT_MODEL` 只填写已经在百炼完成训练并部署的模型 ID。代码负责准备 ChatML 数据、读取部署模型并做四组对照，不会在本机自动创建训练任务或上传含个人信息的数据。训练完成后把模型 ID 写入 `.env`，再使用锁定测试集生成报告。

从已审核的 DPO 对中抽取锁定评测集时，使用独立输出文件；抽取结果仅用于评测，不得回流 SFT 或 DPO 训练：

```bash
conda activate before-choosing-demo
python scripts/extract_locked_trial_cases.py \
  --input datasets/trial_agent/v1/sol_dpo_pairs.local.jsonl \
  --output datasets/trial_agent/eval/locked_cases.v1.jsonl \
  --per-task 2
```

Windows PowerShell：

```powershell
conda activate before-choosing-demo
python .\scripts\extract_locked_trial_cases.py `
  --input .\datasets\trial_agent\v1\sol_dpo_pairs.local.jsonl `
  --output .\datasets\trial_agent\eval\locked_cases.v1.jsonl `
  --per-task 2
```

该锁定集当前为待人工抽检版本，正式提交评测前应复核其 `gold` 标签并固定数据版本。

### 教师生成、异常抽检与数据导出

案例生成和教师评价均提供本地 SQLite 缓存。相同任务、质量级别、模型和 Prompt 版本只产生一次 API 请求；`--dry-run` 只检查输入和打印计划，不调用百炼。

1. 生成文本作答案例候选。默认覆盖 12 个固定任务和 L1–L5 五个质量级别；可以先只生成一个任务验证格式：

```bash
conda activate before-choosing-demo
python scripts/generate_trial_case_inputs.py --task M-02 --levels L3,L4 --dry-run
python scripts/generate_trial_case_inputs.py --task M-02 --levels L3,L4
# 扩充为 DPO 候选集：12 个任务 × 5 个质量级别 × 2 个独立变体
python scripts/generate_trial_case_inputs.py --levels L1,L2,L3,L4,L5 --variants 2 --resume
```

Windows PowerShell：

```powershell
conda activate before-choosing-demo
python .\scripts\generate_trial_case_inputs.py --task M-02 --levels L3,L4 --dry-run
python .\scripts\generate_trial_case_inputs.py --task M-02 --levels L3,L4
# 扩充为 DPO 候选集
python .\scripts\generate_trial_case_inputs.py --levels L1,L2,L3,L4,L5 --variants 2 --resume
```

真实生成最多按“任务数 × 质量级别数 × 变体数”调用一次文本模型；默认使用 `TRIAL_TEACHER_MODEL`。
`--variants 2` 时目标为 120 条案例，覆盖每个任务的 L1–L5 和两种独立作答；已有 case_id 会被
`--resume` 跳过。生成答案不等同于金标准，不能直接训练，必须经过教师评价、证据校验和重复筛选。

2. 使用教师模型生成评价并执行确定性校验。教师调用一次；命中缺失维度、权重不一致、无效证据引用、重复标签或非高置信度时，才升级到 `TRIAL_REVIEW_MODEL`：

```bash
python scripts/build_trial_teacher_labels.py \
  --input datasets/trial_agent/v1/case_inputs.local.jsonl \
  --dry-run
python scripts/build_trial_teacher_labels.py \
  --input datasets/trial_agent/v1/case_inputs.local.jsonl
```

Windows PowerShell：

```powershell
python .\scripts\build_trial_teacher_labels.py `
  --input .\datasets\trial_agent\v1\case_inputs.local.jsonl `
  --dry-run
python .\scripts\build_trial_teacher_labels.py `
  --input .\datasets\trial_agent\v1\case_inputs.local.jsonl
```

`--dry-run` 不会写入伪造评价；正式运行会写入本地 `teacher_labels.local.jsonl`，并保留每条校验状态、证据覆盖率、原因码、模型和请求指纹。`silver_auto` 是可进入候选 SFT 的自动通过记录，`needs_review` 必须由人工抽检并在元数据中标记 `human_reviewed=true`。

3. 生成基础评价并准备逐案例对比包。基础评价使用 `TrialAgent.BASE_SYSTEM_PROMPT`，只作为 DPO
   的候选拒答；强化评价使用上一步的 `qwen3-vl-plus` 结果。两种评价分别缓存，单条案例最多各调用一次：

```bash
python scripts/build_trial_teacher_labels.py \
  --input datasets/trial_agent/v1/case_inputs.local.jsonl \
  --output datasets/trial_agent/v1/baseline_labels.local.jsonl \
  --prompt-variant base --model qwen-plus --prompt-version trial-base-v1 \
  --no-review --resume
python scripts/prepare_sol_pair_packets.py \
  --cases datasets/trial_agent/v1/case_inputs.local.jsonl \
  --teacher datasets/trial_agent/v1/teacher_labels.generated.local.jsonl \
  --baseline datasets/trial_agent/v1/baseline_labels.local.jsonl \
  --sol-sample-count 7 --selection-seed 20260829
```

Windows PowerShell：

```powershell
python .\scripts\build_trial_teacher_labels.py `
  --input .\datasets\trial_agent\v1\case_inputs.local.jsonl `
  --output .\datasets\trial_agent\v1\baseline_labels.local.jsonl `
  --prompt-variant base --model qwen-plus --prompt-version trial-base-v1 `
  --no-review --resume
python .\scripts\prepare_sol_pair_packets.py `
  --cases .\datasets\trial_agent\v1\case_inputs.local.jsonl `
  --teacher .\datasets\trial_agent\v1\teacher_labels.generated.local.jsonl `
  --baseline .\datasets\trial_agent\v1\baseline_labels.local.jsonl `
  --sol-sample-count 7 --selection-seed 20260829
```

每个 packet 对应一条案例的 baseline/teacher 评价对。将剩余 packet 分别交给一个独立的
复核子任务；固定随机抽取 7 条使用 `gpt-5.6-sol / high`，其余使用
`gpt-5.6-luna / max`。抽样结果保存在 `review_assignments.local.json`，续跑不会重新分配。
结果写入同名的
`datasets/trial_agent/v1/sol_review_results.local/` 文件，不能修改 packet。结果必须包含
`case_id`、`pair_valid`、`chosen_source`、`rejected_source`、`enhanced_evaluation`、`rationale`、
`review_model` 和 `reasoning_effort`；
`enhanced_evaluation` 只能引用该 packet 的证据目录。每个 DPO 对只使用一个子任务复核，不能把
一个子任务的结论批量套用到其他案例。

所有子任务结果写完后，离线校验并导出显式 DPO 对：

```bash
python scripts/finalize_sol_pair_reviews.py \
  --cases datasets/trial_agent/v1/case_inputs.local.jsonl \
  --teacher datasets/trial_agent/v1/teacher_labels.generated.local.jsonl \
  --baseline datasets/trial_agent/v1/baseline_labels.local.jsonl \
  --reviews-dir datasets/trial_agent/v1/sol_review_results.local
```

只有证据引用有效、chosen/rejected 有真实差异且独立复核明确判定 `pair_valid=true` 的记录会进入
`sol_dpo_pairs.local.jsonl`；该文件已经是百炼可直接上传的 DPO JSONL，只包含 `messages`、
`chosen`、`rejected` 三个顶层字段，且后两者均为 `role=assistant` 的消息对象。`case_id`、
`task_id` 和复核元数据保留在 `sol_pair_reviews.local.jsonl`，不会混入上传文件。其余记录保留在
`sol_pair_reviews.local.jsonl` 供人工抽检，不会被静默转成负样本。需要内部追踪记录时，可额外传入
`--internal-dpo-output datasets/trial_agent/v1/sol_dpo_pairs.internal.local.jsonl`。目标数据量为
120 条案例和尽可能多的有效对，实际 DPO 数量以校验结果为准。

4. 导出 SFT 候选。默认只导出 `silver_auto`、`human_approved`、`gold` 和 `approved`；需要纳入已人工审核的异常记录时显式加 `--include-needs-review`：

```bash
python scripts/export_trial_teacher_dataset.py \
  --input datasets/trial_agent/v1/teacher_labels.local.jsonl \
  --sft-output datasets/trial_agent/v1/teacher_sft.local.jsonl
python scripts/build_trial_sft_dataset.py \
  --input datasets/trial_agent/v1/teacher_sft.local.jsonl \
  --output-dir datasets/trial_agent/v1/generated
```

DPO 不从单条教师输出自动制造拒答样本。只有提供人工审核的 `chosen_evaluation` 与 `rejected_evaluation` pair 时，才会导出可直接上传百炼的 DPO ChatML 数据，避免把未审核的模型错误当作负样本：

```bash
python scripts/export_trial_teacher_dataset.py \
  --input datasets/trial_agent/v1/teacher_labels.local.jsonl \
  --dpo-input datasets/trial_agent/v1/reviewed_pairs.local.jsonl \
  --dpo-output datasets/trial_agent/v1/teacher_dpo.local.jsonl
```

以上命令均为本地数据处理；只有不带 `--dry-run` 的案例生成和教师评价命令会调用百炼。多模态材料不会被送入 SFT/DPO 数据，图片内容必须先由运行时 Qwen-VL 转成带来源定位的文本证据。

## 本地岗位 RAG 与职业推演

职业探索页只读取已确认能力卡。后端将能力卡内容与本地岗位资料组合成检索词，使用百炼 `qwen3.7-text-embedding` 生成查询向量，在本地 SQLite 向量表中做余弦检索，再把带引用 ID 的片段交给 Qwen 生成结构化推演。当前 `RAG_RETRIEVER_MODE=vector` 是扩展评测集上的默认策略；`adaptive` 是成本受控的实验模式，仅在向量 Top1 与 Top2 间隔低于 `RAG_ADAPTIVE_MARGIN` 时调用 `qwen3-rerank`，并固定保留向量 Top-K 候选集合，防止重排降低召回；`hybrid` 用于复现 SQLite FTS5 + 向量融合 + 重排的固定对照链路。前端不会直接请求百炼，也不会接触 API Key。

模型选择依据：岗位资料是文本，使用 `qwen3.7-text-embedding`，避免引入视觉模型或本地模型文件；重排使用同属 Qwen 系列的 `qwen3-rerank`，保持与比赛技术基础一致。若当前百炼工作空间仍提供 `gte-rerank-v2`，可将 `BAILIAN_RERANK_MODEL` 改为该值，接口协议不变。

知识库资料来自项目提供的 `公共RAG知识库` 解压内容，已按文档登记 `document_id`、资料级别和来源说明。当前岗位文档属于公开资料交叉归纳稿，原始 JD 链接尚待补齐，因此界面会显示资料级别和来源提示，不将归纳稿当作官方岗位结论。

首次安装或更新 Markdown 资料后，先建立本地 FTS5 索引：

```bash
conda run -n before-choosing-demo python -m app.knowledge.indexer
```

Windows PowerShell 使用同一条命令。服务启动时会检查文件指纹并自动建立缺失的 FTS5 索引。

然后使用百炼 Embedding 建立本地向量索引。该步骤首次会按批次调用 Embedding API，之后按资料指纹、模型、维度和内容摘要复用已有向量，不会重复生成未变化的片段：

```bash
conda run -n before-choosing-demo python scripts/build_vector_index.py
```

Windows PowerShell：

```powershell
conda run -n before-choosing-demo python .\scripts\build_vector_index.py
```

向量索引写入 `KNOWLEDGE_DB_PATH` 指定的本机 SQLite 文件，未建立向量索引时系统仍可使用 FTS5；远端 Embedding 暂时不可用时，职业推演会保留确定性的本地检索结果。扩展 26 条查询后，纯向量 Hit@5=100%、MRR@5=94.6%，融合 + 重排 Hit@5=96.2%、MRR@5=87.8%。自适应实验仅对 12 条低置信度查询调用重排，结果 Hit@5=100%、MRR@5=94.6%，与纯向量持平且没有稳定增益，因此正式默认仍选择纯向量；自适应模式保留为可复现的低置信度路由方案。

职业推演接口：

- `POST /api/v1/career/recommendations`：提交 1–4 个已确认能力卡 ID，返回 AI 产品经理路径摘要、支持性判断、未知项、动态选择的下一任务和本地引用片段。

### 检索精度对比

仓库内置小型标注集 `scripts/rag_eval_cases.json`，以资料章节为目标，比较改造前 FTS5 与改造后 Embedding + Rerank 的 `Hit@5` 和 `MRR@5`。默认只运行 FTS5，不调用模型：

```bash
conda run -n before-choosing-demo python scripts/evaluate_rag.py
```

向量索引建立完成且需要真实对比时，显式增加 `--live`。评测集每条查询只调用必要的查询向量和重排，不会重新生成文档向量：

```bash
conda run -n before-choosing-demo python scripts/evaluate_rag.py --live
```

评测集规模较小，用于本地回归和方向性比较；正式材料中的结论应同时记录资料版本、模型版本和调用日期。

验证自适应路由时使用独立脚本。默认离线读取缓存，不产生费用；增加 `--live` 后，仅对低置信度查询调用配置的 Rerank，并将结果写入本地缓存：

```bash
conda run -n before-choosing-demo python scripts/evaluate_adaptive_rag.py
conda run -n before-choosing-demo python scripts/evaluate_adaptive_rag.py --live
```

Windows PowerShell：

```powershell
conda run -n before-choosing-demo python .\scripts\evaluate_adaptive_rag.py
conda run -n before-choosing-demo python .\scripts\evaluate_adaptive_rag.py --live
```

该实验报告写入 `evaluation-results/rag-adaptive-v1/`。自适应路由会保留向量 Top-K 候选集合，并固定向量 Top1，远端重排只能作为次级排序信号；若重排不可用或置信度不足，自动回退到向量结果。

### 统一评测与成本报告

各评测脚本只读取已有数据和预测结果，不会隐式调用百炼。先运行 RAG 与多模态离线评测，再将结果合并为一份报告：

```bash
conda run -n before-choosing-demo python scripts/evaluate_rag.py
conda run -n before-choosing-demo python scripts/evaluate_multimodal.py \
  --cases datasets/multimodal/cases.jsonl \
  --predictions datasets/multimodal/predictions.jsonl
conda run -n before-choosing-demo python scripts/build_unified_evaluation_report.py \
  --trial-report evaluation-results/trial-agent-v1/report.json \
  --rag-report evaluation-results/rag-v1/report.json \
  --multimodal-report evaluation-results/multimodal-v1/report.json
```

报告写入 `evaluation-results/`（已忽略）。统一报告包含 TrialAgent 四组对照、RAG Hit/MRR、多模态页码与区域定位指标，以及正式模式的请求数、模型调用数、Token、平均延迟和按参数估算的费用。只有明确增加 `--live` 时才会产生检索 API 调用。

真实多模态 OCR 对比将 PDF 页面渲染为图片，仅把图片发送给百炼视觉模型；PDF 文字层只在本地作为金标计算字符相似度。下列命令会对每个模型、每页发起一次真实调用：

```bash
conda run -n before-choosing-demo python scripts/evaluate_multimodal_ocr.py \
  --pdf /absolute/path/to/document.pdf --pages 8 --model qwen-vl-ocr \
  --output-dir evaluation-results/multimodal-ocr-qwen-vl-ocr-v1
conda run -n before-choosing-demo python scripts/evaluate_multimodal_ocr.py \
  --pdf /absolute/path/to/document.pdf --pages 8 --model qwen3-vl-plus \
  --output-dir evaluation-results/multimodal-ocr-qwen3-vl-plus-v1
```

完成 TrialAgent、RAG 和两组 OCR 实验后，使用以下离线命令汇总证据；该命令不调用模型：

```bash
conda run -n before-choosing-demo python scripts/build_experiment_evidence_report.py \
  --trial-report evaluation-results/trial-agent-locked-v1/aggregate/report.json \
  --rag-report evaluation-results/rag-live-v1/report.json \
  --multimodal-report evaluation-results/multimodal-ocr-qwen-vl-ocr-v1/report.json \
  --multimodal-report evaluation-results/multimodal-ocr-qwen3-vl-plus-v1/report.json
```

报告同时记录样本量和结论边界。小型锁定集用于回归和方向性对比，不替代更大规模的人工审核测试集。

### P0 核心技术实验

四组 P0 实验覆盖校验链变异测试、RAG 四组消融、并发幂等测试和 TrialAgent 证据敏感性测试。前三组完全读取本地资产，不调用百炼：

macOS：

```bash
conda activate before-choosing-demo
python scripts/run_p0_experiments.py
```

Windows PowerShell：

```powershell
conda activate before-choosing-demo
python .\scripts\run_p0_experiments.py
```

执行 TrialAgent 证据敏感性实验时增加 `--live-sensitivity`。该实验对 12 个固定任务分别评价原始作答与删证据作答，形成 24 个唯一有效模型结果；结果按请求指纹写入本地缓存，重复执行不会再次付费调用：

macOS：

```bash
python scripts/run_p0_experiments.py --live-sensitivity --concurrency 3
```

Windows PowerShell：

```powershell
python .\scripts\run_p0_experiments.py --live-sensitivity --concurrency 3
```

报告写入 `evaluation-results/p0-experiments-v1/report.md` 和 `report.json`。`evaluation-results/` 已加入 Git 忽略规则，不提交模型输出、缓存和实验中间产物。

### 正式模式审计日志

前端正式模式请求携带 `X-App-Mode: use`，后端中间件记录每个非健康检查请求的路径、状态、延迟和请求标识；模型网关同时记录 Qwen、Qwen-VL、Embedding、Rerank 的模型名、延迟与返回的 Token 用量。用户答案、上传材料和提示词正文不写入日志。演示模式不写入正式审计记录。

查看用量摘要：

```bash
curl -H "X-App-Mode: use" http://localhost:8000/api/v1/audit/usage
curl -H "X-App-Mode: use" http://localhost:8000/api/v1/audit/events?limit=50
```

页面点击和表单变更由前端以 `ui_action` 事件记录，后端接口调用由 `http_request` 事件记录，二者共享客户端请求标识，可用于复盘完整操作链路。

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
