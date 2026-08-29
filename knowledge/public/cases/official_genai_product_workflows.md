# 官方生成式 AI 产品工作流案例（跨企业摘要）

整理日期：2026-08-30。本文档把官方产品文档、工程文章、系统卡和平台指南转换为“可以做什么、怎样交付、如何验证”的案例。它不是产品广告合集；没有明确工作流、限制或验证方式的内容不作为强证据。

## OpenAI：把 Evals 当作产品需求的可执行版本

来源：OpenAI，https://openai.com/index/evals-drive-next-chapter-of-ai/ 。官方方法建议从业务目标出发，定义流程中的关键决策点、成功标准和要避免的行为，再用真实场景和边界样例形成评测集。

### PM 交付物

需求文档应与评测集互相对应：每个核心用户目标有样例输入、预期行为、评分维度和失败处理。没有可执行评测的“可靠性要求”无法进入验收。

## OpenAI：Agent 评测需要记录 Harness

来源：OpenAI，https://openai.com/index/trustworthy-third-party-evaluations-foundations/ 。官方建议在 Agent 评测中记录模型、工具、提示、记忆、重试、环境、预算和护栏，因为 Harness 会显著影响结果。

### 任务拆解

可将案例转成“比较两个 Agent 方案”的任务，要求固定模型、工具和预算，报告能力、护栏和比较结论，并检查 reward hacking、污染、拒答和无效问题。

## OpenAI：系统卡作为发布证据

来源：OpenAI Operator System Card，https://openai.com/index/operator-system-card/ 。系统卡将模型数据、风险识别、红队、缓解措施、标准拒答评测和限制集中记录。

### 发布证据包

高风险功能上线时，产品经理应能提供风险清单、攻击样例、评测结果、缓解措施、已知限制和后续监控，而不是只给成功 Demo。

## Google Cloud：Judge Model 与人工真值

来源：Google Cloud，https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/evaluate-judge-model 。Google 的模型评测支持以人工评分作为 ground truth，比较点式和成对评分，并用 Judge Model 扩展评测规模。

### 评价规则

Judge 不是自动正确答案。需要先定义人工评分标准、检查 Judge 与人工的一致性，再使用自动评分；领域高风险任务要保留人工复核。

## Google Cloud：Agent 与 RAG 的分层评测

来源：Google Cloud，https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/evaluate 。Agent 评测可通过评测任务查看模型输出、指标和元数据。

### 产品诊断

把失败定位为查询理解、检索、工具调用、生成或用户控制问题，再决定改数据、改提示、改工具还是改模型。不要把所有失败都归因给模型。

## Databricks：从 Trace 到生产监控

来源：Databricks，https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/ 。MLflow 3 的 GenAI 评测覆盖开发、测试和生产，支持 Trace、内置/自定义 Scorer、LLM Judge、领域专家反馈和版本追踪。

### 工作流案例

一个 RAG/Agent 产品应记录每一步输入、输出、工具和耗时，在离线集上评测，再对线上采样请求持续评分。这样才能知道改动改善了哪一类错误，是否引入成本或延迟回归。

## Databricks：RAG 质量的四个面

来源：Databricks，https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/implementation/step-5-debug-retrieval-quality 。官方建议同时关注业务 KPI、利益相关者反馈、质量指标和生产日志/Trace，并把正确性、延迟、相关性、安全和用户反馈纳入迭代。

### PM 交付物

应有评测集、质量指标定义、Trace 字段、线上采样、告警和反馈闭环。只有一个离线准确率无法支撑 RAG 上线判断。

## Databricks：检索质量与生成质量分开

来源：Databricks，https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/quality-overview 。数据组成、解析、切片、元数据、Embedding、查询改写、召回数和重排都会影响检索质量；生成质量还需单独评估。

### 诊断任务

对一个失败回答，先判断正确证据是否被召回；若没有，修数据管线或检索；若有但答案错误，再查生成、提示或引用约束。

## Anthropic：Transparency Hub 与系统卡

来源：Anthropic Transparency Hub，https://www.anthropic.com/transparency 。官方公开模型训练信息、发布版本、能力/安全评测摘要和系统卡入口。

### 证据使用

Transparency Hub 适合证明官方对模型能力、风险和版本的公开说明；不能把公开摘要当作完整内部评测，也不能用它推断未公开的产品承诺。

## Anthropic：研究、模型行为与 Safeguards

来源：Anthropic Careers，https://www.anthropic.com/careers/jobs 。官方岗位按研究产品、模型行为、Claude Code 性能、Human Data 和 Safeguards 等方向组织。

### 产品工作流

模型行为产品的基本循环是定义行为目标、构造样例、评测、分析边界、实施缓解、再次验证和发布。该循环可映射到任务库的评测和安全任务。

## Meta：Llama 的数据、训练与推理生产链

来源：Meta Engineering，https://engineering.fb.com/2024/08/21/production-engineering/bringing-llama-3-to-life/ 。Llama 生产链包含数据准备、规模化训练、推理并行和异构硬件部署。

### 交付启示

模型产品不能只在研究指标上做决策，还要考虑数据新鲜度、服务规模、上下文窗口、硬件利用率、延迟和成本。PM 要能把这些约束写进版本目标。

## Meta：Responsible Use Guide 的生命周期视角

来源：Meta AI，https://ai.meta.com/llama/responsible-use-guide/ 。官方指南按微调、部署、系统安全和 Responsible Agents 提供开发资源。

### 安全闭环

把安全控制放入数据、模型、工具、应用和运营各阶段，并保留风险、护栏、人工升级、监控和更新记录。安全不是一段发布文案。

## Adobe：Firefly 企业内容供应链

来源：Adobe，https://www.adobe.com/products/firefly/enterprise.html 。Firefly Enterprise 组合创意模型、Creative Cloud、Express、API、Custom Models 和品牌智能，覆盖从生产到分发的内容供应链。

### 业务案例

营销团队需要批量生成、定制、本地化和审批大量资产。PM 要设计模板、品牌规则、权限、人工审查、版本和 Content Credentials，并用上市速度、资产一致性、复用率和成本评估效果。

## Adobe：Firefly Graph 的可复用工作流

来源：Adobe Blog，https://business.adobe.com/blog/meet-firefly-graph 。官方案例以电商耳机活动为例，用节点式工作流组合产品合成、背景生成、文案和品牌约束。

### 从单次生成到工作流资产

真正可积累的不是一张图片，而是可复制的工作流、规则和决策。任务评价应看流程是否可复用、可审计、可调参和可交接。

## Adobe：合作模型的统一治理

来源：Adobe HelpX，https://helpx.adobe.com/firefly/web/work-with-enterprise-features/partner-models-in-firefly-creative-production-for-enterprise.html 。官方资料说明合作模型在 Firefly 的统一工作流中受访问控制和企业权限管理。

### 多模型选择任务

要求候选人根据任务质量、版权、数据、成本和延迟选择模型，并写出准入、回退、审计和管理员权限，而不是简单追求模型数量。

## Apple：端侧与私有云的任务路由

来源：Apple，https://security.apple.com/documentation/private-cloud-compute/ 。Apple Intelligence 在可能时本地处理，复杂任务再使用 Private Cloud Compute，并以无状态、不可定向和可验证透明度为安全要求。

### 适用场景

该案例适合训练“模型路由与隐私”判断：先按任务复杂度、数据敏感度、设备能力、延迟和成本做决策，再设计用户提示、失败回退和验证证据。

## Apple：App Intents 连接应用动作

来源：Apple Developer，https://developer.apple.com/documentation/appintents 。应用用结构化的 App Intent、Entity 和参数让系统发现内容并执行动作。

### Agent 工具设计

App Intent 可以视为受控工具接口。产品经理要定义动作边界、参数、权限、幂等性、确认与撤销，并用真实用户表达测试意图解析。

## 阿里云百炼：知识库的必定调用与智能调用

来源：阿里云帮助中心，https://help.aliyun.com/zh/model-studio/using-the-knowledge-base 。百炼支持知识库必定调用或智能调用，并提供知识库过滤、相似度阈值和权重配置。

### 路由设计

高频专业问答可以固定召回；开放对话需要根据意图决定是否检索。调参要记录阈值、召回量、过滤结果和答案变化，不能凭一次命中测试下结论。

## 阿里云百炼：RAG 效果优化

来源：阿里云帮助中心，https://help.aliyun.com/zh/model-studio/rag-optimization/ 。官方流程把 RAG 拆为索引、召回和生成，并建议先建立覆盖核心真实场景的评测基线，再逐项调整。

### 本地知识库的对应关系

本项目的 FTS5、Embedding、Rerank、切片、来源和回归评测分别对应这三个阶段；任何索引改动都应与固定评测集一起记录。

## 百度千帆：知识库作为 Agent 数据基础

来源：百度智能云，https://cloud.baidu.com/doc/qianfan/s/Imh4stpo0 。官方知识库支持多种文件、解析策略、切片、向量化、检索和命中测试，并用于 Agent 知识问答。

### 知识库运营

企业知识库除了建库，还要处理文件版本、目录、解析失败、切片策略、命中测试和删除同步。适合转化为“知识库上线与维护”任务。

## 百度千帆：多智能体动态规划

来源：百度智能云，https://cloud.baidu.com/doc/qianfan/s/Nmh4stsf3 。官方多智能体支持规划 Agent 拆分复杂任务，再分配给子 Agent，并提供预览、调试和任务过程查看。

### Agent 任务证据

评测要检查任务拆分是否合理、子 Agent 选择是否正确、工具调用是否安全、失败是否可恢复，以及最终答案是否引用了正确的中间结果。

## 百度千帆：深度研究 Agent

来源：百度智能云，https://cloud.baidu.com/doc/qianfan/s/Smky0b2sm 。官方产品支持联网搜索、上传文件、多步骤检索和带引用的结构化报告。

### 研究型交付物

研究 Agent 的质量不只在最终文字，还在检索覆盖、来源可信度、引用对应关系、步骤可追踪和成本控制。可将其转成研究计划、来源台账和引用审查任务。

## 火山方舟：模型推理、评测与精调全流程

来源：火山引擎，https://www.volcengine.com/docs/82379/66619f8df281250274ef4f88?lang=zh 。方舟提供模型推理、评测、精调和安全互信能力。

### 选型与迭代

产品经理应明确何时做 Prompt、RAG、精调或换模型，并用同一评测集比较质量、稳定性、成本和时延，避免把一次演示提升误判为长期收益。

## 火山方舟：模型评测任务

来源：火山引擎，https://www.volcengine.com/docs/82379/1150782?lang=zh 。官方模型评测支持从模型广场、模型仓库或精调模块选择评测对象，建立可比较的质量结果。

### 评测资产

模型选择、数据集版本、评测维度、阈值和报告都应纳入版本库。模型替换后要重跑相同基线，并保留失败样例。

## 华为云盘古：RAG 与 Agent 平台入口

来源：华为云盘古产品资料，https://support.huaweicloud.com/productdesc-pangulm/%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D%E2%80%94%E8%BD%AC%E6%B5%8B%E7%89%B9%E6%80%A7-pdf.pdf 。官方资料列出 Pangu-RAG、Pangu-Agent 等能力入口。

### 资料边界

该 PDF 用于证明产品能力类别和版本，不足以推断所有企业客户效果；需要把性能、时延和质量结论与具体测试数据分开。

## 跨企业工作流的通用结构

可以把官方案例统一为：场景定义 → 数据/上下文准备 → 模型或工具选择 → 交互和权限 → 评测与安全 → 发布与监控 → 反馈和版本迭代。检索时保留企业与产品特有差异，通用结构只作为辅助索引。
