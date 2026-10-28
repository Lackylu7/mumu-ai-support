from __future__ import annotations

from dataclasses import dataclass

from .models import (
    InterpretationBasis,
    InterpretationStatus,
    KnowledgeStatus,
    RiskLevel,
    Sensitivity,
    SourceStatus,
    TestCategory,
    TestReviewStatus,
)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    field_type: str
    options: tuple[str, ...] = ()

    def as_dws_field(self) -> dict[str, object]:
        result: dict[str, object] = {"fieldName": self.name, "type": self.field_type}
        if self.options:
            result["config"] = {"options": [{"name": option} for option in self.options]}
        elif self.field_type == "date":
            result["config"] = {"formatter": "YYYY-MM-DD HH:mm:ss"}
        return result


@dataclass(frozen=True)
class TableSpec:
    name: str
    primary_field: str
    fields: tuple[FieldSpec, ...]


def _field(name: str, field_type: str = "text", *options: str) -> FieldSpec:
    return FieldSpec(name, field_type, tuple(options))


BUSINESS_DOMAINS = (
    "教练运营",
    "员工合伙制",
    "门店运营",
    "人力与行政",
    "客户服务与品牌",
    "公司通用",
)

TABLE_SPECS = (
    TableSpec(
        "知识源台账",
        "来源编号",
        (
            _field("来源编号"),
            _field("资料标题"),
            _field("一级业务域", "singleSelect", *BUSINESS_DOMAINS),
            _field("来源类型", "singleSelect", "钉钉文档", "附件", "AI表格", "人工规则"),
            _field("来源链接", "url"),
            _field("来源节点ID"),
            _field("来源负责人"),
            _field("原文更新时间", "date"),
            _field("生效时间", "date"),
            _field("失效时间", "date"),
            _field("原文版本"),
            _field("接入状态", "singleSelect", *(status.value for status in SourceStatus)),
            _field("原文快照", "richText"),
            _field("审核备注", "richText"),
        ),
    ),
    TableSpec(
        "知识条目",
        "知识编号",
        (
            _field("知识编号"),
            _field("一级业务域", "singleSelect", *BUSINESS_DOMAINS),
            _field("二级主题", "singleSelect"),
            _field("用户意图"),
            _field("常见问法", "richText"),
            _field("标准答案", "richText"),
            _field("成立条件", "richText"),
            _field("例外情形", "richText"),
            _field("办理步骤", "richText"),
            _field("来源编号"),
            _field("来源章节"),
            _field("生效时间", "date"),
            _field("失效时间", "date"),
            _field("风险等级", "singleSelect", *(level.value for level in RiskLevel)),
            _field("信息可见范围", "singleSelect", *(level.value for level in Sensitivity)),
            _field("回答边界", "singleSelect", "可直接回答", "需核验个人数据", "需确认当前规则", "需负责人判断"),
            _field("发布状态", "singleSelect", *(status.value for status in KnowledgeStatus)),
            _field("入库确认人"),
            _field("适用制度版本"),
        ),
    ),
    TableSpec(
        "制度解读",
        "解读编号",
        (
            _field("解读编号"),
            _field("关联知识编号", "richText"),
            _field("一级业务域", "singleSelect", *BUSINESS_DOMAINS),
            _field("适用对象"),
            _field("员工关切", "richText"),
            _field("背后逻辑", "richText"),
            _field("对员工的意义", "richText"),
            _field("员工成本与限制", "richText"),
            _field("推荐表达", "richText"),
            _field("表达边界", "richText"),
            _field("依据等级", "singleSelect", *(basis.value for basis in InterpretationBasis)),
            _field("解读来源"),
            _field("来源章节"),
            _field("发布状态", "singleSelect", *(status.value for status in InterpretationStatus)),
            _field("关联制度版本"),
        ),
    ),
    TableSpec(
        "知识缺口",
        "问题摘要",
        (
            _field("问题摘要"),
            _field("案件编号", "richText"),
            _field("一级业务域", "singleSelect", *BUSINESS_DOMAINS),
            _field("二级主题", "singleSelect"),
            _field("出现次数", "number"),
            _field("首次出现", "date"),
            _field("最近出现", "date"),
            _field("处理状态", "singleSelect", "待处理", "处理中", "已补充", "不采纳"),
            _field("负责人"),
            _field("补充知识编号"),
            _field("处理备注", "richText"),
            _field("原消息ID"),
            _field("提问人ID"),
            _field("提问人姓名"),
            _field("案件群ID"),
            _field("案件群名"),
            _field("建群时间", "date"),
            _field("结案触发人ID"),
            _field("结案触发人姓名"),
            _field("结案时间", "date"),
            _field("自动来源编号"),
            _field("负责人答复", "richText"),
            _field("提交沉淀", "checkbox"),
        ),
    ),
    TableSpec(
        "木木测试题",
        "测试编号",
        (
            _field("测试编号"),
            _field("类别", "singleSelect", *(category.value for category in TestCategory)),
            _field("问题", "richText"),
            _field("必要上下文", "richText"),
            _field("标准答案要点", "richText"),
            _field("禁止内容", "richText"),
            _field("依据知识编号", "richText"),
            _field("风险等级", "singleSelect", *(level.value for level in RiskLevel)),
            _field("审核状态", "singleSelect", *(status.value for status in TestReviewStatus)),
            _field("最近结果", "singleSelect", "未执行", "通过", "失败", "需复核"),
            _field("最近执行时间", "date"),
            _field("结果备注", "richText"),
        ),
    ),
    TableSpec(
        "发布记录",
        "发布编号",
        (
            _field("发布编号"),
            _field("一级业务域", "singleSelect", *BUSINESS_DOMAINS),
            _field("发布批次版本"),
            _field("条目数量", "number"),
            _field("发布时间", "date"),
            _field("审核人"),
            _field("内容摘要", "richText"),
            _field("知识编号清单", "richText"),
            _field("回读校验摘要", "richText"),
        ),
    ),
)


def get_table_spec(name: str) -> TableSpec:
    for table in TABLE_SPECS:
        if table.name == name:
            return table
    raise KeyError(f"未知数据表: {name}")
