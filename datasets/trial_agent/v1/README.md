# TrialAgent SFT 与统一评测数据

本目录只保存数据格式、标注边界和可复现的生成方式。教师模型输出只能作为银标候选；经过确定性校验和人工抽检后，才可作为训练标签。流程不从模型历史输出自动反推任务内容，也不新增固定任务。

## 数据来源与边界

- 任务定义、五步 Schema、材料、事件、Rubric 和 L1–L5 锚点来自 `app/tasks/catalog.py`、`app/tasks/source_content.py` 和 `app/tasks/evaluation_rules.py` 中的固定 12 个任务。
- 每条进入训练集的记录需要核对：答案是否完成、每个 Rubric 分项分、Observed Level、可引用证据 ID，以及能力卡应用证据。`silver_auto` 记录可先进入抽检队列，命中异常规则的 `needs_review` 记录必须在元数据中标记 `human_reviewed=true`。
- 训练数据只保留已脱敏的任务上下文和用户作答。不得把真实姓名、联系方式、公司内部资料或百炼密钥写入 JSONL。
- 多模态模型只用于运行时提取图片或扫描 PDF 的文本证据；SFT/DPO 训练集只包含文本任务上下文、作答和评价 JSON，不对多模态模型做后训练。
- `test_locked` 只用于最终对照，不上传到 SFT 训练任务。建议至少锁定两个任务，并在报告中记录任务 ID。

## 输入格式

输入为人工审核 JSONL，每行必须包含 `case_id`、固定任务 `task_id` 和 Bailian ChatML `messages`。最后一条消息必须是合法 JSON 的 `assistant` 标注：

```json
{"case_id":"m02-001","task_id":"M-02","messages":[{"role":"system","content":"只输出符合任务 Rubric 的 JSON。"},{"role":"user","content":"任务上下文与用户作答……"},{"role":"assistant","content":"{\"summary\":\"……\",\"dimensions\":[]}"}],"metadata":{"source":"human_review"}}
```

脚本会检查任务是否在固定任务库中、`assistant` 是否位于末尾、标注是否为 JSON 对象，以及 `case_id` 是否重复。脚本不生成标签。

## 生成 SFT 分片

### 教师候选链路（可选）

先用固定 12 个任务定义生成文本作答案例，再由教师模型评价。两步都使用 `TRIAL_TEACHER_CACHE_PATH` 指定的本地 SQLite 缓存，重复请求不会重复付费：

```bash
conda activate before-choosing-demo
python scripts/generate_trial_case_inputs.py --task M-02 --levels L3,L4 --dry-run
python scripts/generate_trial_case_inputs.py --task M-02 --levels L3,L4
python scripts/build_trial_teacher_labels.py \
  --input datasets/trial_agent/v1/case_inputs.local.jsonl \
  --dry-run
python scripts/build_trial_teacher_labels.py \
  --input datasets/trial_agent/v1/case_inputs.local.jsonl
python scripts/export_trial_teacher_dataset.py \
  --input datasets/trial_agent/v1/teacher_labels.local.jsonl \
  --sft-output datasets/trial_agent/v1/teacher_sft.local.jsonl
```

`--dry-run` 不调用百炼，也不生成可被误认为答案或评价的内容。正式生成的案例是候选输入，正式评价结果会记录模型、Prompt 版本、请求指纹、证据覆盖率和异常原因码。重复案例和重复标签会被保留为审计记录，但不会自动进入 `silver_auto`。

异常记录完成线下审核后，可在对应 JSONL 的 `metadata` 中加入 `"human_reviewed": true`，并将 `validation.status` 设为 `human_approved`，再执行 SFT 导出。

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

### DPO pair 边界

DPO 需要人工明确的一对 `chosen_evaluation` / `rejected_evaluation`。单条教师输出没有足够信息构成拒答样本，导出脚本会跳过缺少 pair 或 chosen/rejected 完全相同的记录，不自动拼接负样本。导出文件标记为 `dpo-chatml-v1`，上传百炼前按当期控制台的数据格式再次核对。

## 标注规模建议

第一轮建议由同一套标注规范形成约 300–500 条训练记录、60–80 条验证记录，并锁定 120 条测试记录；至少有两个任务只出现在锁定测试集中。规模不足时先保证跨任务、跨证据覆盖，不用复制相似答案填充数量。
