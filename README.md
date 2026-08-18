# 选择之前 API

第一条真实链路：经历输入 → ProfileAgent → 百炼 Qwen → 候选证据卡。

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
# 在 .env 填入 DASHSCOPE_API_KEY
uvicorn app.main:app --reload --port 8000
```

健康检查：`GET http://localhost:8000/api/v1/health`

候选卡接口：`POST http://localhost:8000/api/v1/profile/proposals`

```json
{
  "experience_text": "我在校园项目中访谈用户并根据反馈调整了方案，最后完成了可用原型。",
  "target_role": "AI Native 产品经理"
}
```

缺少 `DASHSCOPE_API_KEY` 或 Qwen 返回格式不合法时，接口会返回明确错误，不返回伪造结果。
