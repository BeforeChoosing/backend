# 官方生成式 AI 评测、治理与交付方法（摘要）

整理日期：2026-08-30。本文档用于补充公共 RAG 的 Rubric、工作台证据和索引评测规范。所有结论均为官方资料的短摘要；指标和阈值需要结合本地任务，不直接照搬供应商默认值。

## 评测先写清楚要证明什么

来源：OpenAI，https://openai.com/index/trustworthy-third-party-evaluations-foundations/ 。评测报告应说明要测试的是能力、护栏还是系统比较，并公开足够的任务、系统配置、预算、诱导方式和有效性检查。

### Rubric 规则

每个评测项先写 claim，再写输入、环境、工具、预期行为、评分方式和证据位置。一个分数不能同时代表能力、可靠性和安全性。

## 评测环境是结果的一部分

来源：OpenAI，https://openai.com/index/trustworthy-third-party-evaluations-foundations/ 。Agent 的模型、工具、记忆、重试、控制逻辑和环境会改变表现，报告必须固定或记录这些 Harness 条件。

### 本地实验要求

RAG 实验要保存 Embedding 模型、索引版本、查询改写、召回数量、重排模型、提示版本和是否允许远端调用；否则无法判断提升来自数据、检索链还是偶然的环境变化。

## 评测集应代表真实任务和边界

来源：OpenAI，https://openai.com/index/evals-drive-next-chapter-of-ai/ ；Databricks，https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/implementation/step-5-debug-retrieval-quality 。官方建议使用真实场景，并加入罕见但高代价的边界样例。

### 查询集分层

本项目评测集应同时包含事实型、比较型、教程型、分析型、多跳型、歧义型、错别字、越权/注入和来源追溯查询。每次扩充只新增查询，不删除旧回归样例。

## 人工真值优先于盲目使用 Judge

来源：Google Cloud，https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/evaluate-judge-model 。Judge Model 的分数需要与人工评分或成对选择进行校准，并检查平衡准确率、混淆矩阵等一致性。

### 评分器治理

记录评分器版本、提示、参考答案、人工样本和一致性结果。对关键任务保留人工复核；Judge 结果只能作为证据的一部分。

## RAG 检索和生成要分层诊断

来源：Databricks，https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/quality-overview 。检索质量受数据组成、解析、切片、元数据、Embedding、查询改写、召回数量和重排影响；生成质量是另一个层面。

### 失败归因

若正确片段未被召回，归因为数据或检索；若片段已召回但答案错，归因为生成、提示或引用；若答案正确但来源不支持，归因为证据绑定。每类失败要有不同修复动作。

## 线上监控要和离线评测同一套指标

来源：Databricks，https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/ 。官方建议在开发、测试和生产使用一致的 Trace、Scorer 和人工反馈，并持续记录版本。

### 生产指标

至少记录请求量、错误率、延迟、Token/成本、相关性、有据性、安全通过率、用户显式反馈和追问/放弃等隐式信号。线上采样不能替代固定回归集。

## RAG 的来源和版本是答案的一部分

来源：阿里云百炼，https://help.aliyun.com/zh/model-studio/rag-knowledge-retrieval 。官方检索服务支持查询改写、混合检索、排序、相似度阈值、召回数量和过滤配置。

### 本地实现规则

每个片段要保存来源 URL、章节、版本、检索日期、可信度和内容哈希；查询结果要能回到片段 ID 和来源文档。来源缺失时降低置信度，不用通用常识填空。

## 知识库切片策略需要可解释

来源：百度千帆知识库，https://cloud.baidu.com/doc/qianfan/s/Imh4stpo0 。官方支持多种解析和切片策略，并提供知识库命中测试。

### 切片验收

切片要保留标题路径和必要上下文；表格、图片、公式和长文要记录解析方式；调整切片策略后必须比较旧/新查询集的召回变化，并保存失败样例。

## 多知识域要用意图路由

来源：阿里云百炼，https://help.aliyun.com/zh/model-studio/rag-knowledge-retrieval 。官方支持多知识库联合检索和知识库路由。

### 本地知识域

本项目至少区分岗位、任务、产品案例、技术方法、评测安全和工作台证据。先识别查询意图，再在相关域内召回，避免岗位 JD 与技术规范互相污染。

## 高风险动作必须有用户控制

来源：Apple PCC 安全指南，https://security.apple.com/documentation/private-cloud-compute/ ；Meta Responsible Use Guide，https://ai.meta.com/llama/responsible-use-guide/ 。官方资料分别强调隐私边界、验证和安全部署。

### 交付要求

对修改外部状态、处理敏感数据或影响高风险决策的 Agent，必须设计最小权限、用户确认、撤销、人工升级、日志审查和失败回退。只写“加 Guardrail”不构成可验证方案。

## 数据最小化与用户资料隔离

来源：Apple Private Cloud Compute，https://security.apple.com/documentation/private-cloud-compute/ 。PCC 资料将无状态计算、无特权运行时和可验证透明度作为核心要求。

### 公共 RAG 边界

公共知识库只收录公开、可引用和已脱敏资料。用户原始答案、联系方式、上传文件和内部运行日志不能进入公共索引；需要保留的实验诊断应先脱敏并放在本地结果目录。

## 多语言与地区版本不能混为一谈

来源：Apple Machine Learning Research，https://machinelearning.apple.com/research/apple-foundation-models-2025-updates 。官方资料说明评测数据会按语言和地区扩展，并由本地语言专家修订。

### 评测切片

记录语言、地区、输入模态、格式和用户群体。翻译后的查询不等于真实本地任务；关键功能要有原生语言样例和失败类型。

## 安全评测要覆盖注入、滥用和过度拒答

来源：Meta Llama Responsible Use Guide，https://ai.meta.com/llama/responsible-use-guide/ ；OpenAI System Card，https://openai.com/index/operator-system-card/ 。官方资料涉及红队、风险缓解、拒答和安全评估。

### 评分维度

将危险任务拒答、正常任务不过度拒答、工具越权、提示注入、敏感信息泄露和输出可操作性分别评分。一个“安全通过率”不能覆盖全部风险。

## 证据强度要和结论范围匹配

官方岗位页可以证明岗位职责，官方产品文档可以证明能力和使用方式，System Card 可以证明公布的评测与限制，案例宣传页只能证明产品方向或公开案例。不能用一个来源支撑超出它范围的结论。

## 版本、撤回和失效处理

每次官方页面变化都新建或更新版本，不覆盖旧事实；记录 `published_at`、`retrieved_at`、`status`、`version` 和 `supersedes`。已关闭职位、预览功能和延期功能默认降低新鲜度与可信度。

## RAG 实验报告应包含的最小字段

报告至少写：语料版本、文档数、片段数、向量数、Embedding 模型与维度、FTS5/纯向量/RAG v2 配置、评测集版本、Hit@1、Hit@5、MRR、按知识域切片结果、失败查询、来源正确率、调用次数和成本估计。

## 扩充后的发布门槛

新增资料只有在来源字段完整、无用户资料、重复率可控、索引数量一致、旧回归集不退化且新知识域查询达到预设门槛时才进入 active。未满足条件的资料保留在待复核清单，不进入生产检索。
