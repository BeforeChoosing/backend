# 选择之前 API

后端负责第一条真实链路：

```text
经历输入 → ProfileAgent → 阿里云百炼 Qwen → 结构化候选证据卡
```

后端使用 Conda 管理 Python 环境，当前提供 FastAPI API、Qwen 网关和 ProfileAgent。

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

## A-02 最小试路任务

当前 Demo 只接入任务库中的 `A-02｜这个 Agent 为什么总是失败？`，不动态生成其他试路题。任务内容、8 个 Bad Case、归因层、事件和评价维度均来自 CoachAgent 任务库；指标与案例在任务库中明确标注为模拟试路材料。

试路接口：

- `GET /api/v1/trial/tasks/A-02`：读取固定任务和前台材料。
- `POST /api/v1/trial/sessions`：创建本机作答会话。
- `GET /api/v1/trial/sessions/{session_id}`：恢复会话。
- `PUT /api/v1/trial/sessions/{session_id}/answer`：保存结构化作答。
- `POST /api/v1/trial/sessions/{session_id}/event`：触发中途事件。
- `POST /api/v1/trial/sessions/{session_id}/submit`：提交给 Qwen 按任务 Rubric 评价，并生成 `Observed Evidence`。

单次任务只形成 `Observed Evidence`，不直接生成岗位胜任力等级或企业认证结论。

## 本地画像持久化

用户点击“加入能力库”后，已确认卡片会写入本机 SQLite 文件。`PROFILE_DB_PATH` 用于指定文件位置，默认值为 `profile.db`；该文件已加入 Git 忽略规则，不会提交到仓库。

画像接口：

- `GET /api/v1/profile/cards`：读取已确认卡片，页面刷新后恢复。
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
