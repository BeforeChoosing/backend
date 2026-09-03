# Apple AI 评测与交付证据（官方资料摘要）

整理日期：2026-08-30。本文件把 Apple 的公开技术、开发者和安全资料转换为可用于任务库与 Rubric 的证据规则。内容是短摘要，原始事实以链接页面和页面版本为准。

## 评测框架：从 Prompt 到可发布功能

来源：Apple Developer WWDC26 平台说明，https://developer.apple.com/videos/play/wwdc2026/112/ 。Apple 在 Foundation Models 相关工具链中提供 Evaluations，用于测试 Prompt 和验证智能功能是否稳定；开发工具还支持观察和调试模型行为。

### 可评分证据

合格作答应包含评测目标、样例输入、预期输出、评分标准、失败分类、版本号和回归门槛。只展示一条成功样例，不能证明功能可靠。

## Agent 评测的多步轨迹

来源：Apple Developer WWDC26，https://developer.apple.com/videos/play/wwdc2026/242/ 。Agent 可能经历多轮上下文、工具调用、模型交接和错误恢复，因此不能只看最终文本质量。

### 评测维度

至少记录任务完成率、工具选择正确率、参数正确率、恢复成功率、上下文长度、延迟和用户确认次数。对改变外部状态的动作，还要增加权限和撤销测试。

## RAG 评测：召回、引用与答案分开

来源：Apple Developer WWDC26 平台说明，https://developer.apple.com/videos/play/wwdc2026/112/ 。应用内 RAG 将检索、模型生成和应用数据边界结合起来。

### Rubric 结构

将评分拆成检索相关性、引用支持度、答案事实一致性、数据权限正确性和删除/更新一致性五项。检索命中但引用不支持答案时，不能记为完整成功。

## 端侧模型的性能证据包

来源：Apple Machine Learning Research，https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025 。端侧模型需要在 Apple silicon 的资源约束下运行。

### 必填指标

性能报告至少包含设备型号、系统版本、模型版本、量化/压缩配置、输入长度、首 token 延迟、总耗时、峰值内存、功耗或电量影响、任务质量和失败回退。不同设备的平均值不能混为一个“端侧性能”。

## 本地与私有云路由的验收

来源：Apple Private Cloud Compute 安全指南，https://security.apple.com/documentation/private-cloud-compute/ 。PCC 的安全要求包括无状态计算、无特权运行时访问、不可定向攻击和可验证透明度。

### 验收清单

发布前检查：路由条件是否可解释；发送字段是否最小化；设备是否验证节点身份；请求数据是否在响应后清除；日志是否会泄露原文；失败时是否回到本地或请求用户确认；软件版本是否能被独立复核。

## 安全评测：从威胁到缓解

来源：Apple Security Research，https://security.apple.com/blog/pcc-security-research/ 。Apple 公开了安全指南、虚拟研究环境、透明日志和部分源代码，以便外部研究者验证隐私承诺。

### Rubric 评分锚点

L1 只能列出风险名词；L2 能说明攻击前提和影响；L3 能提出对应护栏、日志和回滚；L4 能给出可复现的测试步骤、证据位置和残余风险。没有复现或验证方式的“安全设计”只能算方案假设。

## 多语言与无障碍交付

来源：Apple Machine Learning Research，https://machinelearning.apple.com/research/apple-foundation-models-2025-updates ；Apple Newsroom，https://www.apple.com/newsroom/2026/05/apple-unveils-new-accessibility-features-and-updates-with-apple-intelligence/ 。Apple 公开资料强调多语言评测和面向残障用户的自然语言、视觉描述与阅读辅助场景。

### 证据要求

必须按语言、地区、输入模态和用户群体分层报告；同时记录误识别、不可用场景、人工替代和用户控制。只在英文或普通用户样本上通过，不能推断所有地区和辅助场景通过。

## Apple 案例如何映射到产品任务

推荐将案例转换为以下任务：选择模型路由；设计 App Intent schema；设计 Agent 状态机；建立 Prompt 回归集；写 RAG 引用与删除策略；制定端侧性能预算；完成 PCC 威胁模型；准备无障碍验收；编写版本化发布清单。

## 来源可信度与时效

Apple Developer 文档、WWDC 视频和 Security Research 作为一级官方来源；Machine Learning Research 技术报告作为一级研究来源；Newsroom 作为功能发布和产品方向来源。对 `preview`、延期、地区限制和已废弃 API 必须单独标记状态，不能把历史描述作为当前承诺。
