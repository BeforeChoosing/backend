# TrialAgent 统一评测集

评测集每行一个 JSON 对象，输入必须来自固定任务库，参考标签由人工审核填写。建议将 `cases.jsonl` 保存在本机或私有存储，锁定测试集不与 SFT 训练数据混用。

```json
{"case_id":"m02-001","task_id":"M-02","answer":{"step_answers":{"scenarios":"……","failures":"……","budget":"……","threshold":"……","event":"……"},"event_decision":"调整","event_response":"……"},"gold":{"dimensions":{"模型评测":82,"用户洞察":72},"observed_level":"L3","evidence_refs":["answer:scenarios","answer:event"]}}
```

预测结果由外部运行或人工导入，每行格式如下；`valid_evidence_refs` 必须是服务端为该案例生成的证据目录，不接受模型自报的来源：

```json
{"case_id":"m02-001","arm":"prompt_hardened","evaluation":{"summary":"……","dimensions":[],"observed_level":"L3","confidence":"中","strengths":[],"gaps":[],"next_step":"……"},"valid_evidence_refs":["answer:scenarios","answer:event"],"api_calls":1,"latency_ms":1234}
```

允许的 `arm`：`base_qwen`、`prompt_hardened`、`sft`、`sft_validator`。`sft_validator` 只在证据校验被触发时记录校验触发率，不把校验结果当作额外分数。

仓库中的 `cases.example.jsonl` 和 `predictions.example.jsonl` 只用于验证报告脚本的格式链路，标注状态为 `example`，不能作为模型精度结论。正式对比需要替换为人工审核的锁定测试集和四组真实预测结果。
