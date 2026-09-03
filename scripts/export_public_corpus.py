"""Export the reviewed local task sources into Markdown RAG documents.

The runtime task catalog is the single source of truth for task text. This
script creates stable, human-reviewable Markdown snapshots; it never reads
user profiles or network data and is safe to run offline.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tasks.catalog import TASK_CATALOG
from app.tasks.evaluation_rules import TASK_EVALUATION_RULES


PUBLIC = ROOT / "knowledge" / "public"


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(line.strip() for line in lines if line.strip()) + "\n", encoding="utf-8")


def _task_catalog() -> list[str]:
    lines = [
        "# AI 产品经理职业试路任务库（代码快照）",
        "本文件由 `app/tasks/catalog.py` 与 `source_content.py` 导出。任务结构、材料、事件、步骤和能力挑战均为公共试路设计；其中业务数字和案例明确标记为模拟材料。",
        "## 统一使用说明",
        "每个任务包含一个主能力、两项支持能力、五个可观察步骤、一个动态事件和三条 Coach 提示。作答时应引用材料，区分事实、假设与待验证项，并记录是否使用 Coach。",
    ]
    for task_id, task in TASK_CATALOG.items():
        lines.extend(
            [
                f"## {task.id}｜{task.title}",
                f"{task.subtitle}",
                "### 任务定位",
                f"角色：{task.role}；角色线：{task.role_type}；工作阶段：{task.work_stage}；难度：{task.difficulty}；建议时长：{task.estimated_minutes}。",
                f"主能力：{task.primary_skill}；支持能力：{'、'.join(task.supporting_skills)}。",
                f"背景：{task.background}",
                f"目标：{task.goal}",
                "### 约束",
                "；".join(task.constraints),
                "### 材料",
            ]
        )
        for material in task.materials:
            simulated = "模拟材料" if material.is_simulated else "公开材料"
            lines.extend(
                [
                    f"#### {material.id}｜{material.title}",
                    f"类型：{material.kind}；性质：{simulated}。{material.content}",
                ]
            )
        lines.append("### 五步作答")
        for index, step in enumerate(task.steps, start=1):
            lines.extend(
                [
                    f"#### {index}. {step.title}",
                    f"输入方式：{step.input_mode}。要求：{step.instruction} 约束：{step.constraint}",
                ]
            )
        lines.extend(
            [
                "### 动态事件",
                f"事件角色：{task.event.actor}。事件：{task.event.message}。处理要求：{task.event.instruction}",
                "### Coach 提示",
                "；".join(task.coach_prompts),
                "### 能力挑战",
            ]
        )
        for challenge in task.ability_challenges:
            lines.extend(
                [
                    f"#### {challenge.id}｜{challenge.title}",
                    f"场景：{challenge.scenario} 目标能力：{'、'.join(challenge.target_skills)} 参考行为：{challenge.reference_behavior}",
                ]
            )
    return lines


def _evaluation_rules() -> list[str]:
    lines = [
        "# Rubric 与评价规则（代码快照）",
        "本文件由 `app/tasks/evaluation_rules.py` 导出。Rubric 用于观察作答行为，不把一次任务总分直接等同于能力等级。",
        "## 统一证据规则",
        "一个任务设置一个主能力（约 35%–50%）和 2–4 个支持维度；评价必须引用最终交付物、证据使用、决策过程、动态事件响应和 Coach 依赖。",
        "观察证据只记录在任务完成后的 Observed Evidence 中。同一能力至少需要两个相互独立的证据才能形成 Current Level；高置信结论应有至少四条证据并覆盖至少三个 Task Atom。",
        "Prompt 或 Coach 的使用需要记录，但不能简单扣分；应区分独立完成、借助提示完成和仍无法完成。模拟数据必须显式标注，不得写成真实业务数据。",
        "## 等级解释",
        "L1 表示主要凭偏好或套模板；L2 能引用少量材料但推理不稳定；L3 能完成证据驱动的闭环判断；L4 能处理竞争性假设、约束和证据缺口；L5 能建立可迁移的判断原则并在新约束下主动取舍。",
    ]
    for task_id, rules in TASK_EVALUATION_RULES.items():
        task = TASK_CATALOG[task_id]
        lines.extend(
            [
                f"## {task_id}｜{task.title}",
                "### 维度与权重",
            ]
        )
        for dimension, weight, behavior in rules["rubric"]:  # type: ignore[index]
            lines.extend(
                [
                    f"#### {dimension}（{weight}%）",
                    behavior,
                ]
            )
        lines.append("### 等级锚点")
        for level in ("L1", "L2", "L3", "L4", "L5"):
            lines.extend([f"#### {level}", str(rules["anchors"][level])])  # type: ignore[index]
    return lines


def _workbench_rules() -> list[str]:
    return [
        "# 工作台与证据评价规范",
        "本文件是对本地产品说明书、用户画像 PRD、技术交接规则和公开方法资料的去敏整理。它只保留产品流程与评价规范，不纳入用户画像、个人答案或其他敏感资料。",
        "来源文件：`选择之前_产品说明书_v2_卡牌工作台版.html`、`选择之前_PRD_用户画像阶段.md`、`选择之前_P8-P18_技术报告.html`、`CoachAgent_职业试路任务生成与评价逻辑_产品技术交接版_v1.0.docx`；整理版本：workbench-rules-v1.0；整理日期：2026-08-29。",
        "## 工作台主流程",
        "工作台分为资料确认、能力卡确认、任务试路和证据回顾四个阶段。推荐岗位必须同时展示依据、未知项和下一步验证任务；推荐不是录用结论。",
        "用户可确认、拒绝或标记不确定的能力卡。公共岗位资料只用于抽取岗位相关能力与工作任务，个人资料应最小化保存，并与公共知识库隔离。",
        "## 证据链",
        "每条能力结论应能回到任务、步骤、材料或事件；引用必须保留文档、章节和来源定位。最终交付物之外，还要记录证据使用、推理路径、动态调整和 Coach 依赖。",
        "同一能力需要跨任务或跨 Task Atom 的独立证据才可提升置信度。单个任务可以说明一次观察，不足以单独定义职业等级。",
        "## 任务生成质量门槛",
        "任务生成链路为：岗位模型 → 主能力 → Task Atom → 材料 → 干扰信息 → 现实约束 → 作答 Schema → 动态事件 → Rubric。每一环都要保留来源证据。",
        "质量检查包括真实性、可追溯性、能力对齐、材料充分性、多条合理路径、不过载、12–20 分钟可完成、可评分、结构化作答和模拟数据标记。",
        "## 来源治理",
        "官方招聘页、法规、官方技术文档、公开研究或公开访谈优先；来源按权威性、相关性、时效性和具体性分层。一个岗位不能由单一来源定义，无法核验、需登录或付费的内容不入库。",
        "同义词统一为用户洞察/用户研究、Bad Case/错误归因等；岗位名称、职责、交付物和能力要求分开记录，避免把写作能力误当产品能力。",
        "## 数据与隐私红线",
        "不把模拟数字写成真实业务结果；不把用户原始答案、个人联系方式或未脱敏资料放入公共 RAG；不以强制匹配百分比替代证据判断；支持版本、撤回和失效标记。",
        "## 检索使用建议",
        "先用岗位、工作阶段和能力做查询规划，再以 FTS5 与向量候选合并；对多个意图分别检索后用 RRF/MMR 去重。返回片段必须带来源定位、可信度和版本信息。",
    ]


def _jd_matrix() -> list[str]:
    rows = [
        ("JD-01", "字节跳动／剪映", "截图摘要（用户提供）", "", "核心", "A-"),
        ("JD-02", "腾讯／企业微信 AI 助手", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1880770535693004800", "核心", "A"),
        ("JD-03", "腾讯／游戏 AI", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1897945233740599296", "核心", "A"),
        ("JD-04", "腾讯／元宝 AI 策略", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1925398818564710400", "核心", "A"),
        ("JD-05", "腾讯会议／AI 策略", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1942177600126451712", "核心", "A"),
        ("JD-06", "微信妙剪／AI 产品", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1951113659853963264", "核心", "A"),
        ("JD-07", "微信输入法／AI 产品", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1978467627152072704", "核心", "A"),
        ("JD-08", "企业微信／AI Bot", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=1985253844522782720", "核心", "A"),
        ("JD-09", "腾讯／AI 生成游戏", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2011285787706019840", "核心", "A"),
        ("JD-10", "WeGame／AI 应用", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2018950192145006592", "核心", "A"),
        ("JD-11", "腾讯云／AI 供应链", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2026121516504215552", "核心", "A"),
        ("JD-12", "腾讯云／AI 异构计算", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2034117546277498880", "核心", "A"),
        ("JD-13", "腾讯云／MaaS 高级产品", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2034975730101809152", "核心", "A"),
        ("JD-14", "腾讯会议／ASR", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2036621556322562048", "核心", "A"),
        ("JD-15", "企业微信／AI 表格", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2038501839850336256", "核心", "A"),
        ("JD-16", "腾讯会议／AI 应用体验", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2046083747635425280", "核心", "A"),
        ("JD-17", "腾讯云／AI 语音", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2046718048437694464", "核心", "A"),
        ("JD-18", "腾讯视频／AI 产品", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2046845261342470144", "核心", "A"),
        ("JD-19", "腾讯证券／AI 应用体验", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2052527703940313088", "核心", "A"),
        ("JD-20", "企业微信／微信互通", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2054462480285085696", "核心", "A"),
        ("JD-21", "腾讯／AI 协作工具", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2061001589497442304", "核心", "A"),
        ("JD-22", "微信读书／AI 产品", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2068897868235849728", "核心", "A"),
        ("JD-23", "腾讯安全／AI 代码安全", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2070679023687745536", "核心", "A"),
        ("JD-24", "腾讯／支付高级产品", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2072623740965007360", "补充", "A"),
        ("JD-25", "腾讯／混元 3D AIGC", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2074695803712225280", "核心", "A"),
        ("JD-26", "腾讯／光子 AI 数据平台", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2077246089718837248", "核心", "A"),
        ("JD-27", "腾讯／AI 策略游戏与电竞", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2077297592525570048", "核心", "A"),
        ("JD-28", "腾讯 QQ-Agent", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2077347119940939776", "核心", "A"),
        ("JD-29", "腾讯 CodeBuddy Agent", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2077642272568160256", "核心", "A"),
        ("JD-30", "企业微信／AI 文档", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2082723377528877056", "核心", "A"),
        ("JD-31", "腾讯／设计 Agent Ardot", "官方招聘 API", "https://careers.tencent.com/jobdesc.html?postId=2086346849417080832", "核心", "A"),
        ("JD-32", "美图／海外高级 AI 应用产品", "官方招聘页", "https://hr.meitu.com/jobSocial/00e89cbb-8ea9-4f51-86fe-ff7424c8", "核心", "A"),
        ("JD-33", "美图／资深 AI 产品", "官方招聘页", "https://hr.meitu.com/jobSocial/0ab81319-9e11-4659-a644-af5a36a8bc0e", "核心", "A"),
        ("JD-34", "美图／AI 短剧工具", "官方招聘页", "https://hr.meitu.com/jobSocial/1d1f67f8-bfa4-45f0-bd7f-d4c5a8d0071b", "核心", "A"),
        ("JD-35", "美图／AI 用户画像平台", "官方招聘页", "https://hr.meitu.com/jobSocial/21506eb9-b152-46b4-b511-d7207230e309", "核心", "A"),
        ("JD-36", "美图／AI 用户画像平台（重复）", "官方招聘页", "https://hr.meitu.com/jobSocial/312868d3-42ab-483c-942c-e115f718de92", "重复，排除", "A"),
        ("JD-37", "美图／AI 影像商业化", "官方招聘页", "https://hr.meitu.com/jobSocial/f52bf59d-601f-4578-8ba9-a485f01f05d3", "核心", "A"),
        ("JD-38", "百度／AI 产品（校招）", "官方招聘页", "https://talent.baidu.com/jobs/detail/GRADUATE/cb79f6d4-b39e-4e4e-907e-d4d39d4c3f80", "核心", "A"),
    ]
    lines = [
        "# 现有岗位资料来源矩阵",
        "本文件从 `CoachAgent_产品经理资料库_v3_AI工作链版.xlsx` 的 JD 样本与来源台账整理而来。它记录来源与去重状态，不把岗位原文整段复制进公共库。",
        "## 统一能力标签",
        "岗位样本统一映射到用户洞察、方案与交互、AI 产品化、跨团队落地、数据驱动、模型评测、创新趋势和商业意识；映射结果用于检索候选与任务推荐，不能替代岗位原文。",
    ]
    current_company = ""
    for jd_id, company, source_type, url, use, grade in rows:
        company_name = company.split("／", 1)[0]
        if company_name != current_company:
            current_company = company_name
            lines.append(f"## {current_company}")
        lines.extend(
            [
                f"### {jd_id}｜{company}",
                f"来源类型：{source_type}；使用状态：{use}；来源等级：{grade}。",
                f"来源 URL：{url or '无可核验 URL，仅保留用户提供截图摘要，不作为官方原文引用。'}",
            ]
        )
    lines.extend(
        [
            "## 去重与纳入规则",
            "同一岗位的重复链接只保留一条；截图摘要与官方原文分开标识；只有职责、要求和来源可核验的记录进入岗位能力归纳。",
            "来源台账中的 16 条截图/经验材料和 1 条未核验访谈仅作为待核验线索，不直接进入高可信岗位结论。",
        ]
    )
    return lines


def main() -> None:
    outputs = {
        PUBLIC / "tasks" / "task_catalog.md": _task_catalog(),
        PUBLIC / "tasks" / "evaluation_rules.md": _evaluation_rules(),
        PUBLIC / "workbench" / "workbench_evidence_rules.md": _workbench_rules(),
        PUBLIC / "jobs" / "jd_evidence_matrix.md": _jd_matrix(),
    }
    for path, lines in outputs.items():
        _write(path, lines)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
