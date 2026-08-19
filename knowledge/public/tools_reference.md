# 公共工具文档

> 供 Phase 01 / Phase 02 共用。Phase 03 只用 search_job_requirements。

---

## search_job_requirements

搜索目标岗位的核心能力要求。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `position_name` | string | 是 | 目标岗位名称，如 "AI产品经理"、"用户体验设计师" |

**返回：**

```json
{
  "status": "success",
  "position": "AI产品经理",
  "rag": "job_facts",
  "results": [
    {
      "heading": "需求分析与转化",
      "content": "深入调研用户需求与市场动态，将模糊业务痛点转化为清晰的AI产品需求方案。来源：猎聘-海大集团AI产品经理JD",
      "score": 10
    }
  ],
  "count": 5
}
```

**对应 RAG：** 岗位事实 RAG（`knowledge/jobs/` 目录下的岗位文件）

---

## search_capability_method

搜索能力方法库——追问话术、卡牌格式、证据类型判断规则。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索关键词，如 "追问话术"、"能力卡格式"、"证据类型判断" |

**返回：**

```json
{
  "status": "success",
  "rag": "capability_method",
  "results": [
    {
      "heading": "能力卡格式",
      "content": "每张能力卡用以下格式输出：...",
      "score": 3
    }
  ],
  "count": 2
}
```

**对应 RAG：** 能力方法 RAG（`knowledge/capability_method.md` + `knowledge/job_search_workflow.md`）
