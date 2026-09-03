# AI 产品实践与检索基础（官方资料摘要）

整理日期：2026-08-29。本文件只保留官方方法指南和模型服务文档的短摘要，链接是事实核验入口；不复制受限原文，不把摘要当作标准全文。

## NIST AI Risk Management Framework

来源：NIST 官方 AI RMF 页面 https://www.nist.gov/itl/ai-risk-management-framework 。AI 产品在机会判断、设计、上线和运营中都应持续识别风险、衡量影响并留下治理责任，而不是只在上线前做一次检查。

### 可迁移到任务的判断

把治理拆成 Govern、Map、Measure、Manage 四类动作：明确责任与政策；理解场景、用户和影响；用指标、测试和证据衡量质量与风险；根据结果排序、缓解、监控或停止。任务作答应写出风险、证据缺口、护栏指标和回滚条件。

## NIST Generative AI Profile

来源：NIST 官方公开 PDF https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf 。生成式 AI 需要关注模型输出的可靠性、隐私、偏差、信息完整性、供应链和人机配置等风险。

### 交付物提示

产品交付物不应只有 PRD，还应包含风险登记、评测集或抽样方案、人工复核机制、发布门槛、监控与事件响应。对不可逆或高影响动作，应明确用户确认和降级路径。

## Google People + AI Guidebook

来源：Google PAIR 官方指南 https://pair.withgoogle.com/guidebook-v2/ 。指南强调围绕人的目标设计 AI 系统，并在交互中说明能力边界、置信度、反馈和控制方式。

### 反馈与控制

来源：PAIR Feedback and Control https://pair.withgoogle.com/guidebook-v2/chapter/feedback-controls/ 。AI 输出应允许用户理解、纠正、撤销或继续处理；产品经理需要设计反馈入口、失败恢复和在环分工，而不是把错误留给用户自行承担。

### 数据收集与评测

来源：PAIR Data Collection and Evaluation https://pair.withgoogle.com/guidebook-v2/chapter/data-collection/ 。评测应覆盖真实任务、边界场景和用户反馈，明确收集目的、样本范围、质量标准与迭代闭环；“模型分数提升”不能单独代表用户价值提升。

## Qwen Embedding 服务约束

来源：阿里云百炼官方文本向量 API 文档 https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api 和 Embedding 模型说明 https://help.aliyun.com/zh/model-studio/embedding 。本项目继续使用 `qwen3.7-text-embedding`，本地索引维度固定为 1024，批量请求上限按服务文档与项目配置取 20 条。

### 索引工程要求

文档向量和查询向量必须区分 text_type；向量结果落本地 SQLite，不把原文或用户资料发往外部向量数据库。新增文档时按内容哈希增量生成，未变化片段复用已有向量；模型或维度变化时才整体失效。

## Qwen Rerank 服务

来源：阿里云百炼官方文本重排 API 文档 https://help.aliyun.com/zh/model-studio/text-rerank-api 。Rerank 只接收有限候选，适合在 FTS5 与向量召回后做精排；候选数量、超时、失败回退和成本应记录在检索诊断中。

### 与工作台的结合

对岗位、任务和证据规范的多意图查询，先分别召回再用 RRF/MMR 去重；需要远端重排时保留原始候选顺序、模型名和是否成功，失败则回到确定性排序，不把一次远端评分当成能力结论。

## 使用边界

以上官方资料用于补充风险、交互、评测和检索工程方法，不用于替代岗位 JD、任务 Rubric 或个人能力判断。若链接内容更新，应在 manifest 中增加版本和检索日期，并保留旧版本的撤回记录。
