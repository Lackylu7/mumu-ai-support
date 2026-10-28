from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .case_domain import (
    CaseMessage,
    CaseRecord,
    ExtractedKnowledge,
    knowledge_id_for_case,
    publish_id_for_case,
    source_id_for_case,
)


class CaseRepositoryError(RuntimeError):
    pass


RICH_TEXT_FIELDS = {
    "案件编号",
    "处理备注",
    "原文快照",
    "审核备注",
    "常见问法",
    "标准答案",
    "成立条件",
    "例外情形",
    "办理步骤",
    "内容摘要",
    "知识编号清单",
    "回读校验摘要",
    "负责人答复",
}


def _records(payload: object) -> list[dict[str, object]]:
    value = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _plain_cell(value: object) -> str:
    if isinstance(value, dict):
        for name in ("markdown", "text", "value", "name"):
            candidate = value.get(name)
            if isinstance(candidate, (str, bool)):
                return str(candidate).strip()
    if value is None:
        return ""
    return str(value).strip()


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class CaseRepository:
    def __init__(self, client: Any, manifest: dict[str, object]) -> None:
        self.client = client
        base_id = manifest.get("base_id")
        tables = manifest.get("tables")
        if not isinstance(base_id, str) or not isinstance(tables, dict):
            raise CaseRepositoryError("运行清单缺少 Base 或数据表")
        self.base_id = base_id
        self.tables: dict[str, tuple[str, dict[str, str]]] = {}
        for table_name in ("知识缺口", "知识源台账", "知识条目", "发布记录"):
            table = tables.get(table_name)
            if not isinstance(table, dict):
                raise CaseRepositoryError(f"运行清单缺少数据表: {table_name}")
            table_id = table.get("table_id")
            fields = table.get("fields")
            if not isinstance(table_id, str) or not isinstance(fields, dict):
                raise CaseRepositoryError(f"运行清单数据表无效: {table_name}")
            clean_fields = {
                name: field_id
                for name, field_id in fields.items()
                if isinstance(name, str) and isinstance(field_id, str) and field_id
            }
            self.tables[table_name] = (table_id, clean_fields)

    def _table(self, table_name: str) -> tuple[str, dict[str, str]]:
        return self.tables[table_name]

    def _all(self, table_name: str) -> list[dict[str, object]]:
        table_id, _ = self._table(table_name)
        return _records(self.client.query_all_records(self.base_id, table_id))

    def _cell(self, table_name: str, record: dict[str, object], name: str) -> str:
        _, fields = self._table(table_name)
        field_id = fields.get(name)
        if not field_id:
            raise CaseRepositoryError(f"运行清单 {table_name} 缺少字段: {name}")
        cells = record.get("cells")
        if not isinstance(cells, dict):
            return ""
        return _plain_cell(cells.get(field_id))

    def _mapped_cells(
        self,
        table_name: str,
        values: dict[str, object],
    ) -> dict[str, object]:
        _, fields = self._table(table_name)
        result: dict[str, object] = {}
        for name, value in values.items():
            field_id = fields.get(name)
            if not field_id:
                raise CaseRepositoryError(f"运行清单 {table_name} 缺少字段: {name}")
            if name in RICH_TEXT_FIELDS:
                result[field_id] = {"markdown": str(value or "")}
            else:
                result[field_id] = value
        return result

    def _write_file(self, table_name: str, records: list[dict[str, object]], *, update: bool) -> None:
        table_id, _ = self._table(table_name)
        with TemporaryDirectory(prefix="mumu-case-") as directory:
            path = Path(directory) / "records.json"
            path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if update:
                self.client.update_records(self.base_id, table_id, path)
            else:
                self.client.create_records(self.base_id, table_id, path)

    def _create(self, table_name: str, values: dict[str, object]) -> None:
        self._write_file(
            table_name,
            [{"cells": self._mapped_cells(table_name, values)}],
            update=False,
        )

    def _update(
        self,
        table_name: str,
        record_id: str,
        values: dict[str, object],
    ) -> None:
        self._write_file(
            table_name,
            [
                {
                    "recordId": record_id,
                    "cells": self._mapped_cells(table_name, values),
                }
            ],
            update=True,
        )

    def _find_raw(
        self,
        table_name: str,
        field_name: str,
        value: str,
    ) -> dict[str, object] | None:
        for record in self._all(table_name):
            if self._cell(table_name, record, field_name) == value:
                return record
        return None

    def _case_from_record(self, record: dict[str, object]) -> CaseRecord:
        get = lambda name: self._cell("知识缺口", record, name)
        created_at = _parse_time(get("首次出现"))
        if created_at is None:
            raise CaseRepositoryError("案件首次出现时间无效")
        return CaseRecord(
            record_id=str(record.get("recordId", record.get("id", ""))),
            case_id=get("案件编号"),
            source_message_id=get("原消息ID"),
            asker_user_id=get("提问人ID"),
            asker_name=get("提问人姓名"),
            question=get("问题摘要"),
            gap_reason=get("处理备注"),
            owner_user_id="",
            owner_name=get("负责人"),
            created_at=created_at,
            business_domain=get("一级业务域") or "公司通用",
            group_id=get("案件群ID"),
            group_name=get("案件群名"),
            group_created_at=_parse_time(get("建群时间")),
            status=get("处理状态") or "处理中",
            knowledge_id=get("补充知识编号"),
            source_id=get("自动来源编号"),
            close_trigger_user_id=get("结案触发人ID"),
            close_trigger_name=get("结案触发人姓名"),
            closed_at=_parse_time(get("结案时间")),
            notes=get("处理备注"),
            responsible_answer=get("负责人答复"),
            submission_requested=get("提交沉淀").strip().lower() == "true",
        )

    def find_by_case_id(self, case_id: str) -> CaseRecord | None:
        raw = self._find_raw("知识缺口", "案件编号", case_id)
        return self._case_from_record(raw) if raw is not None else None

    def find_by_message_id(self, message_id: str) -> CaseRecord | None:
        raw = self._find_raw("知识缺口", "原消息ID", message_id)
        return self._case_from_record(raw) if raw is not None else None

    def find_by_group_id(self, group_id: str) -> CaseRecord | None:
        raw = self._find_raw("知识缺口", "案件群ID", group_id)
        return self._case_from_record(raw) if raw is not None else None

    def list_submitted_cases(self) -> tuple[CaseRecord, ...]:
        submitted: list[CaseRecord] = []
        for raw in self._all("知识缺口"):
            requested = self._cell("知识缺口", raw, "提交沉淀")
            if requested.strip().lower() != "true":
                continue
            case = self._case_from_record(raw)
            if case.submission_requested and case.status != "已补充":
                submitted.append(case)
        return tuple(submitted)

    def list_draft_cases(self) -> tuple[CaseRecord, ...]:
        drafts: list[CaseRecord] = []
        for raw in self._all("知识缺口"):
            status = self._cell("知识缺口", raw, "处理状态")
            knowledge_id = self._cell("知识缺口", raw, "补充知识编号")
            notes = self._cell("知识缺口", raw, "处理备注")
            if (
                status == "已补充"
                or not knowledge_id
                or not notes.startswith("已生成草稿：")
            ):
                continue
            drafts.append(self._case_from_record(raw))
        return tuple(drafts)

    def clear_submission(self, case_id: str) -> CaseRecord:
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        self._update(
            "知识缺口",
            case.record_id,
            {"提交沉淀": False},
        )
        updated = self.find_by_case_id(case_id)
        if updated is None:
            raise CaseRepositoryError("清除提交沉淀状态后回读失败")
        return updated

    def record_submission_error(self, case_id: str, message: str) -> CaseRecord:
        return self.update_case_note(case_id, message)

    def ensure_case(self, case: CaseRecord) -> CaseRecord:
        existing = self.find_by_message_id(case.source_message_id)
        if existing is not None:
            return existing
        now_text = case.created_at.isoformat()
        self._create(
            "知识缺口",
            {
                "问题摘要": case.question,
                "案件编号": case.case_id,
                "一级业务域": case.business_domain,
                "出现次数": 1,
                "首次出现": now_text,
                "最近出现": now_text,
                "处理状态": "处理中",
                "负责人": case.owner_name,
                "处理备注": case.gap_reason,
                "原消息ID": case.source_message_id,
                "提问人ID": case.asker_user_id,
                "提问人姓名": case.asker_name,
            },
        )
        created = self.find_by_case_id(case.case_id)
        if created is None:
            raise CaseRepositoryError("创建案件后回读失败")
        return replace(created, owner_user_id=case.owner_user_id)

    def update_case_group(
        self,
        case_id: str,
        *,
        group_id: str,
        group_name: str,
        created_at: datetime,
    ) -> CaseRecord:
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        self._update(
            "知识缺口",
            case.record_id,
            {
                "案件群ID": group_id,
                "案件群名": group_name,
                "建群时间": created_at.isoformat(),
                "最近出现": created_at.isoformat(),
            },
        )
        updated = self.find_by_case_id(case_id)
        if updated is None:
            raise CaseRepositoryError("更新案件群后回读失败")
        return updated

    def update_case_note(self, case_id: str, note: str) -> CaseRecord:
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        self._update("知识缺口", case.record_id, {"处理备注": note})
        updated = self.find_by_case_id(case_id)
        if updated is None:
            raise CaseRepositoryError("更新案件备注后回读失败")
        return updated

    def mark_intro_sent(self, case_id: str) -> CaseRecord:
        return self.update_case_note(case_id, "案件群已创建，问题简介已发送。")

    def ensure_source(
        self,
        case_id: str,
        candidate: ExtractedKnowledge,
        messages: tuple[CaseMessage, ...],
        now: datetime,
        *,
        status: str = "已发布",
    ) -> str:
        source_id = source_id_for_case(case_id)
        existing = self._find_raw("知识源台账", "来源编号", source_id)
        if (
            existing is not None
            and self._cell("知识源台账", existing, "接入状态") == "已发布"
        ):
            return source_id
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        cited = set(candidate.source_message_ids)
        snapshot = "\n\n".join(
            f"[{item.message_id}] {item.sender_name} {item.created_at.isoformat()}\n{item.text}"
            for item in messages
            if item.message_id in cited
        )
        values = {
            "来源编号": source_id,
            "资料标题": f"木木案件结论 {case.case_id}",
            "一级业务域": candidate.business_domain,
            "来源类型": "人工规则",
            "来源链接": "",
            "来源节点ID": case.group_id,
            "来源负责人": "、".join(candidate.source_authors),
            "原文更新时间": now.isoformat(),
            "生效时间": now.isoformat() if status == "已发布" else "",
            "失效时间": "",
            "原文版本": "1.0",
            "接入状态": status,
            "原文快照": snapshot,
            "审核备注": f"由案件 {case.case_id} 自动提取",
        }
        if existing is None:
            self._create("知识源台账", values)
        else:
            self._update(
                "知识源台账",
                str(existing.get("recordId", existing.get("id", ""))),
                values,
            )
        if self._find_raw("知识源台账", "来源编号", source_id) is None:
            raise CaseRepositoryError("创建来源后回读失败")
        return source_id

    def ensure_knowledge(
        self,
        case_id: str,
        candidate: ExtractedKnowledge,
        *,
        trigger_name: str,
        now: datetime,
        status: str = "已发布",
    ) -> str:
        knowledge_id = knowledge_id_for_case(case_id)
        existing = self._find_raw("知识条目", "知识编号", knowledge_id)
        if (
            existing is not None
            and self._cell("知识条目", existing, "发布状态") == "已发布"
        ):
            return knowledge_id
        source_id = source_id_for_case(case_id)
        values = {
            "知识编号": knowledge_id,
            "一级业务域": candidate.business_domain,
            "用户意图": candidate.canonical_question,
            "常见问法": "\n".join(candidate.common_questions),
            "标准答案": candidate.standard_answer,
            "成立条件": "\n".join(candidate.conditions),
            "例外情形": "\n".join(candidate.exceptions),
            "办理步骤": "\n".join(candidate.steps),
            "来源编号": source_id,
            "来源章节": f"案件 {case_id}",
            "生效时间": now.isoformat() if status == "已发布" else "",
            "失效时间": "",
            "风险等级": candidate.risk_level,
            "信息可见范围": "限内部",
            "发布状态": status,
            "入库确认人": f"木木自动整理（触发人：{trigger_name}）",
            "适用制度版本": "1.0",
        }
        if existing is None:
            self._create("知识条目", values)
        else:
            self._update(
                "知识条目",
                str(existing.get("recordId", existing.get("id", ""))),
                values,
            )
        if self._find_raw("知识条目", "知识编号", knowledge_id) is None:
            raise CaseRepositoryError("创建知识后回读失败")
        return knowledge_id

    def publication_is_valid(self, case_id: str) -> bool:
        source_id = source_id_for_case(case_id)
        knowledge_id = knowledge_id_for_case(case_id)
        source = self._find_raw("知识源台账", "来源编号", source_id)
        knowledge = self._find_raw("知识条目", "知识编号", knowledge_id)
        return bool(
            source is not None
            and knowledge is not None
            and self._cell("知识源台账", source, "接入状态") == "已发布"
            and self._cell("知识条目", knowledge, "发布状态") == "已发布"
            and self._cell("知识条目", knowledge, "来源编号") == source_id
        )

    def ensure_publish_record(
        self,
        case_id: str,
        candidate: ExtractedKnowledge,
        *,
        trigger_name: str,
        now: datetime,
    ) -> str:
        publish_id = publish_id_for_case(case_id)
        if self._find_raw("发布记录", "发布编号", publish_id) is not None:
            return publish_id
        knowledge_id = knowledge_id_for_case(case_id)
        self._create(
            "发布记录",
            {
                "发布编号": publish_id,
                "一级业务域": candidate.business_domain,
                "发布批次版本": "1.0",
                "条目数量": 1,
                "发布时间": now.isoformat(),
                "审核人": f"木木自动入库（触发人：{trigger_name}）",
                "内容摘要": "\n".join(
                    (
                        f"规范问题：{candidate.canonical_question}",
                        f"标准答案：{candidate.standard_answer}",
                    )
                ),
                "知识编号清单": knowledge_id,
                "回读校验摘要": "知识源台账与知识条目均已回读，状态和来源关联校验通过。",
            },
        )
        if self._find_raw("发布记录", "发布编号", publish_id) is None:
            raise CaseRepositoryError("创建发布记录后回读失败")
        return publish_id

    def mark_completed(
        self,
        case_id: str,
        *,
        trigger_user_id: str,
        trigger_name: str,
        now: datetime,
    ) -> CaseRecord:
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        self._update(
            "知识缺口",
            case.record_id,
            {
                "处理状态": "已补充",
                "补充知识编号": knowledge_id_for_case(case_id),
                "自动来源编号": source_id_for_case(case_id),
                "结案触发人ID": trigger_user_id,
                "结案触发人姓名": trigger_name,
                "结案时间": now.isoformat(),
                "处理备注": "负责人答复已发布。",
            },
        )
        updated = self.find_by_case_id(case_id)
        if updated is None:
            raise CaseRepositoryError("结案后回读失败")
        return updated

    def mark_draft(
        self,
        case_id: str,
        *,
        trigger_user_id: str,
        trigger_name: str,
        now: datetime,
        reasons: tuple[str, ...],
    ) -> CaseRecord:
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        reason_text = "；".join(reasons) or "群聊信息不足，已生成知识草稿。"
        self._update(
            "知识缺口",
            case.record_id,
            {
                "处理状态": "处理中",
                "补充知识编号": knowledge_id_for_case(case_id),
                "自动来源编号": source_id_for_case(case_id),
                "结案触发人ID": trigger_user_id,
                "结案触发人姓名": trigger_name,
                "结案时间": now.isoformat(),
                "处理备注": f"已生成草稿：{reason_text}",
            },
        )
        updated = self.find_by_case_id(case_id)
        if updated is None:
            raise CaseRepositoryError("草稿状态更新后回读失败")
        return updated

    def disable_case_knowledge(
        self,
        case_id: str,
        *,
        trigger_user_id: str,
        trigger_name: str,
        now: datetime,
    ) -> CaseRecord:
        knowledge = self._find_raw(
            "知识条目",
            "知识编号",
            knowledge_id_for_case(case_id),
        )
        if knowledge is not None:
            record_id = str(knowledge.get("recordId", knowledge.get("id", "")))
            self._update("知识条目", record_id, {"发布状态": "已停用"})
        case = self.find_by_case_id(case_id)
        if case is None:
            raise CaseRepositoryError("案件不存在")
        self._update(
            "知识缺口",
            case.record_id,
            {
                "处理状态": "处理中",
                "结案触发人ID": trigger_user_id,
                "结案触发人姓名": trigger_name,
                "结案时间": now.isoformat(),
                "处理备注": "自动发布结论被群成员标记有误，等待重新确认。",
            },
        )
        updated = self.find_by_case_id(case_id)
        if updated is None:
            raise CaseRepositoryError("回滚后回读失败")
        return updated
