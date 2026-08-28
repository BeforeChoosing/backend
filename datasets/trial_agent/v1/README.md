# TrialAgent SFT 与统一评测数据

本目录只保存数据格式、标注边界和可复现的生成方式。训练标签必须来自人工审核记录，不从模型历史输出自动反推，也不新增任务内容。

## 数据来源与边界

- 任务定义、五步 Schema、材料、事件、Rubric 和 L1–L5 锚点来自 `app/tasks/catalog.py`、`app/tasks/source_content.py` 和 `app/tasks/evaluation_rules.py` 中的固定 12 个任务。
- 每条记录需要人工核对：答案是否完成、每个 Rubric 分项分、Observed Level、可引用证据 ID，以及能力卡应用证据。
- 训练数据只保留已脱敏的任务上下文和用户作答。不得把真实姓名、联系方式、公司内部资料或百炼密钥写入 JSONL。
- `test_locked` 只用于最终对照，不上传到 SFT 训练任务。建议至少锁定两个任务，并在报告中记录任务 ID。

## 输入格式

输入为人工审核 JSONL，每行必须包含 `case_id`、固定任务 `task_id` 和 Bailian ChatML `messages`。最后一条消息必须是合法 JSON 的 `assistant` 标注：

```json
{"case_id":"m02-001","task_id":"M-02","messages":[{"role":"system","content":"只输出符合任务 Rubric 的 JSON。"},{"role":"user","content":"任务上下文与用户作答……"},{"role":"assistant","content":"{\"summary\":\"……\",\"dimensions\":[]}"}],"metadata":{"source":"human_review"}}
```

脚本会检查任务是否在固定任务库中、`assistant` 是否位于末尾、标注是否为 JSON 对象，以及 `case_id` 是否重复。脚本不生成标签。

## 生成 SFT 分片

macOS/Linux：

```bash
conda activate before-choosing-demo
python scripts/build_trial_sft_dataset.py \
  --input datasets/trial_agent/v1/annotated.jsonl \
  --output-dir datasets/trial_agent/v1/generated \
  --holdout-task M-02 \
  --holdout-task A-02
```

Windows PowerShell：

```powershell
conda activate before-choosing-demo
python .\scripts\build_trial_sft_dataset.py `
  --input .\datasets\trial_agent\v1\annotated.jsonl `
  --output-dir .\datasets\trial_agent\v1\generated `
  --holdout-task M-02 `
  --holdout-task A-02
```

生成的 `train.jsonl` 和 `validation.jsonl` 是可上传百炼 SFT 的 `messages` JSONL；`test_locked.jsonl` 和 `manifest.json` 只用于本地评测。`generated/` 已加入 Git 忽略，避免提交训练产物和临时报告。

## 标注规模建议

第一轮建议由同一套标注规范形成约 300–500 条训练记录、60–80 条验证记录，并锁定 120 条测试记录；至少有两个任务只出现在锁定测试集中。规模不足时先保证跨任务、跨证据覆盖，不用复制相似答案填充数量。
