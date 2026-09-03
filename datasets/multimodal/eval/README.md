# 多模态证据定位评测集

这里存放扫描简历、作品截图和项目页面的人工标注格式。每行一个案例，`materials` 是本案例允许出现的材料标识，`gold` 是人工核对后的证据区域。

```json
{"case_id":"mm-001","materials":["resume","portfolio"],"gold":[{"evidence_id":"resume-result","material_id":"resume","page":1,"bbox":[100,100,500,500],"quote":"负责用户访谈并完成方案迭代"}]}
```

模型预测单独保存，不能把预测结果当作金标准：

```json
{"case_id":"mm-001","items":[{"material_id":"resume","page":1,"bbox":[120,120,480,480],"quote":"负责用户访谈并完成方案迭代","confidence":0.92}],"api_calls":1,"latency_ms":1234}
```

评测只接受人工确认的页码、归一化区域（`bbox` 为 0–1000）和连续文字。匹配条件默认是同一材料、同一页、区域 IoU ≥ 0.5 且文字相似度 ≥ 0.5；报告分别给出页码命中率、定位 IoU、证据精确率/召回率、材料覆盖率和页面覆盖率。

样例仅用于验证脚本链路，不代表 Qwen-VL 的真实精度。macOS/Linux 与 Windows 均在后端仓库根目录运行：

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

报告写入 `evaluation-results/multimodal-v1/`，该目录已加入 `.gitignore`。这个过程不调用百炼，也不会产生费用。
