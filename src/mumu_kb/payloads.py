from __future__ import annotations

from enum import Enum

from .models import InterpretationEntry, KnowledgeEntry, SourceRecord, TestCase
from .schema import FieldSpec, get_table_spec


class ManifestError(ValueError):
    pass


def serialize_multivalue(values: list[str]) -> str:
    return "\n".join(values)


def _field_ids(manifest: dict[str, object], table_name: str) -> dict[str, str]:
    try:
        tables = manifest["tables"]
        table = tables[table_name]
        raw_fields = table["fields"]
    except (KeyError, TypeError) as error:
        raise ManifestError(f"运行清单缺少数据表或字段: {table_name}") from error
    if not isinstance(raw_fields, dict):
        raise ManifestError(f"运行清单字段映射格式错误: {table_name}")
    field_ids = {
        str(name): str(field_id)
        for name, field_id in raw_fields.items()
        if isinstance(name, str) and isinstance(field_id, str) and field_id
    }
    expected = {field.name for field in get_table_spec(table_name).fields}
    missing = sorted(expected - set(field_ids))
    if missing:
        raise ManifestError(f"运行清单 {table_name} 缺少字段: {', '.join(missing)}")
    expected_ids = [field_ids[name] for name in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ManifestError(f"运行清单 {table_name} 包含重复 fieldId")
    return field_ids


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return serialize_multivalue([str(item) for item in value])
    return value


def _cell_value(field: FieldSpec, value: object) -> object:
    normalized = _plain(value)
    if field.field_type == "richText":
        return {"markdown": str(normalized or "")}
    if field.field_type == "url":
        return {"text": str(normalized or ""), "link": str(normalized or "")}
    return normalized


def _map_cells(
    table_name: str,
    values: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    ids = _field_ids(manifest, table_name)
    table = get_table_spec(table_name)
    return {
        ids[field.name]: _cell_value(field, values.get(field.name, ""))
        for field in table.fields
    }


def build_source_cells(
    source: SourceRecord,
    manifest: dict[str, object],
) -> dict[str, object]:
    return _map_cells(
        "知识源台账",
        {
            "来源编号": source.source_id,
            "资料标题": source.title,
            "一级业务域": source.business_domain,
            "来源类型": source.source_type,
            "来源链接": source.source_url,
            "来源节点ID": source.source_node_id,
            "来源负责人": source.source_owner,
            "原文更新时间": source.source_updated_at,
            "生效时间": source.effective_from,
            "失效时间": source.effective_to,
            "原文版本": source.version,
            "接入状态": source.status,
            "原文快照": source.raw_snapshot,
            "审核备注": source.review_notes,
        },
        manifest,
    )


def build_knowledge_cells(
    entry: KnowledgeEntry,
    manifest: dict[str, object],
) -> dict[str, object]:
    return _map_cells(
        "知识条目",
        {
            "知识编号": entry.knowledge_id,
            "一级业务域": entry.business_domain,
            "二级主题": entry.business_topic,
            "用户意图": entry.intent,
            "常见问法": entry.question_variants,
            "标准答案": entry.canonical_answer,
            "成立条件": entry.conditions,
            "例外情形": entry.exceptions,
            "办理步骤": entry.steps,
            "来源编号": entry.source_id,
            "来源章节": entry.source_section,
            "生效时间": entry.effective_from,
            "失效时间": entry.effective_to,
            "风险等级": entry.risk_level,
            "信息可见范围": entry.sensitivity,
            "回答边界": entry.answer_boundary,
            "发布状态": entry.status,
            "入库确认人": entry.reviewer,
            "适用制度版本": entry.version,
        },
        manifest,
    )


def build_interpretation_cells(
    entry: InterpretationEntry,
    manifest: dict[str, object],
) -> dict[str, object]:
    return _map_cells(
        "制度解读",
        {
            "解读编号": entry.interpretation_id,
            "关联知识编号": entry.knowledge_ids,
            "一级业务域": entry.business_domain,
            "适用对象": entry.audience,
            "员工关切": entry.employee_concern,
            "背后逻辑": entry.rationale,
            "对员工的意义": entry.employee_value,
            "员工成本与限制": entry.employee_cost,
            "推荐表达": entry.recommended_response,
            "表达边界": entry.boundaries,
            "依据等级": entry.basis,
            "解读来源": entry.source_label,
            "来源章节": entry.source_section,
            "发布状态": entry.status,
            "关联制度版本": entry.version,
        },
        manifest,
    )


def build_test_case_cells(
    case: TestCase,
    manifest: dict[str, object],
) -> dict[str, object]:
    return _map_cells(
        "木木测试题",
        {
            "测试编号": case.test_id,
            "类别": case.category,
            "问题": case.question,
            "必要上下文": case.context,
            "标准答案要点": case.expected_points,
            "禁止内容": case.forbidden_content,
            "依据知识编号": case.knowledge_ids,
            "风险等级": case.risk_level,
            "审核状态": case.review_status,
            "最近结果": "未执行",
            "最近执行时间": "",
            "结果备注": f"预期动作：{case.expected_action.value}",
        },
        manifest,
    )
