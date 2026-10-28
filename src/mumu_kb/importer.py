from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    KnowledgeEntry,
    KnowledgeStatus,
    SourceRecord,
    TestCase,
    load_json_records,
    record_from_dict,
)
from .payloads import build_knowledge_cells, build_source_cells, build_test_case_cells
from .provision import load_manifest


class ImportValidationError(ValueError):
    pass


class ImportConfirmationRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class ImportBatch:
    table_name: str
    records: tuple[dict[str, object], ...]
    business_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportSummary:
    completed_batches: int
    record_count: int


@dataclass(frozen=True)
class VerificationSummary:
    table_counts: dict[str, int]
    empty_rows: dict[str, int]
    missing_ids: dict[str, tuple[str, ...]]
    duplicate_ids: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not any(self.missing_ids.values()) and not any(self.duplicate_ids.values())


def split_records(
    table_name: str,
    records: list[dict[str, object]],
    business_ids: list[str] | None = None,
) -> list[ImportBatch]:
    ids = business_ids or ["" for _ in records]
    if len(ids) != len(records):
        raise ImportValidationError("记录数量与业务 ID 数量不一致")
    return [
        ImportBatch(
            table_name,
            tuple(records[start : start + 30]),
            tuple(ids[start : start + 30]),
        )
        for start in range(0, len(records), 30)
    ]


def _parse_time(value: str, field_name: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ImportValidationError(f"{field_name} 不是合法 ISO 时间: {value}") from error
    if parsed.tzinfo is None:
        raise ImportValidationError(f"{field_name} 必须包含时区: {value}")
    return parsed


def validate_publishable(entry: KnowledgeEntry, now: datetime) -> None:
    entry.validate()
    if entry.status is not KnowledgeStatus.PUBLISHED:
        return
    if now.tzinfo is None:
        raise ImportValidationError("校验时间必须包含时区")
    effective_from = _parse_time(entry.effective_from, "生效时间")
    effective_to = _parse_time(entry.effective_to, "失效时间")
    if effective_from and now < effective_from:
        raise ImportValidationError(f"已发布知识尚未生效: {entry.knowledge_id}")
    if effective_to and now >= effective_to:
        raise ImportValidationError(f"已发布知识已经失效: {entry.knowledge_id}")


def _read_source(pack_dir: Path) -> SourceRecord:
    source_path = pack_dir / "source.json"
    try:
        raw_source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportValidationError(f"无法读取来源元数据: {source_path}") from error
    if not isinstance(raw_source, dict):
        raise ImportValidationError("source.json 顶层必须是对象")
    try:
        source = record_from_dict(SourceRecord, raw_source)
        snapshot = (pack_dir / "source.md").read_text(encoding="utf-8")
    except (ValueError, OSError) as error:
        raise ImportValidationError(f"来源资料校验失败: {error}") from error
    return replace(source, raw_snapshot=snapshot)


def _validate_pack(
    source: SourceRecord,
    entries: list[KnowledgeEntry],
    cases: list[TestCase],
    now: datetime,
) -> None:
    if source.source_id != "POLICY-S001":
        raise ImportValidationError("首份来源编号必须为 POLICY-S001")
    if len(entries) < 45:
        raise ImportValidationError("原子知识少于 45 条")
    if len(cases) < 60:
        raise ImportValidationError("测试题少于 60 条")
    knowledge_ids = {entry.knowledge_id for entry in entries}
    for entry in entries:
        if entry.source_id != source.source_id:
            raise ImportValidationError(f"知识来源不匹配: {entry.knowledge_id}")
        validate_publishable(entry, now)
        if "24 完" in entry.canonical_answer and entry.status is not KnowledgeStatus.REVIEW:
            raise ImportValidationError(f"歧义知识不得发布: {entry.knowledge_id}")
    for case in cases:
        missing = sorted(set(case.knowledge_ids) - knowledge_ids)
        if missing:
            raise ImportValidationError(
                f"测试题 {case.test_id} 引用不存在知识: {', '.join(missing)}"
            )
        if not case.knowledge_ids and case.expected_action.value != "转人工":
            raise ImportValidationError(f"无依据测试题必须转人工: {case.test_id}")


def build_import_plan(
    pack_dir: Path,
    manifest_path: Path,
    now: datetime,
) -> list[ImportBatch]:
    manifest = load_manifest(manifest_path)
    source = _read_source(pack_dir)
    try:
        entries = load_json_records(pack_dir / "entries.json", KnowledgeEntry)
        cases = load_json_records(pack_dir / "test-cases.json", TestCase)
    except ValueError as error:
        raise ImportValidationError(str(error)) from error
    _validate_pack(source, entries, cases, now)

    batches: list[ImportBatch] = []
    batches.extend(
        split_records(
            "知识源台账",
            [{"cells": build_source_cells(source, manifest)}],
            [source.source_id],
        )
    )
    batches.extend(
        split_records(
            "知识条目",
            [{"cells": build_knowledge_cells(entry, manifest)} for entry in entries],
            [entry.knowledge_id for entry in entries],
        )
    )
    batches.extend(
        split_records(
            "木木测试题",
            [{"cells": build_test_case_cells(case, manifest)} for case in cases],
            [case.test_id for case in cases],
        )
    )
    return batches


def write_plan_files(batches: list[ImportBatch], runtime_dir: Path) -> list[Path]:
    if runtime_dir.resolve().name != ".runtime":
        raise ImportValidationError("导入计划必须生成在 .runtime 目录")
    plan_dir = runtime_dir / "import-plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    table_order = {"知识源台账": 1, "知识条目": 2, "木木测试题": 3}
    table_sequences: dict[str, int] = {}
    paths: list[Path] = []
    all_ids: list[str] = []
    summary_tables: dict[str, dict[str, int]] = {}
    for batch in batches:
        sequence = table_sequences.get(batch.table_name, 0) + 1
        table_sequences[batch.table_name] = sequence
        prefix = table_order[batch.table_name]
        path = plan_dir / f"{prefix:02d}-{batch.table_name}-{sequence:03d}.json"
        path.write_text(
            json.dumps(list(batch.records), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
        all_ids.extend(batch.business_ids)
        table_summary = summary_tables.setdefault(batch.table_name, {"batches": 0, "records": 0})
        table_summary["batches"] += 1
        table_summary["records"] += len(batch.records)
    digest = hashlib.sha256("\n".join(all_ids).encode("utf-8")).hexdigest()
    (plan_dir / "summary.json").write_text(
        json.dumps(
            {"tables": summary_tables, "business_id_sha256": digest},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _table_name_from_path(path: Path) -> str:
    match = re.fullmatch(r"\d{2}-(.+)-\d{3}", path.stem)
    if not match:
        raise ImportValidationError(f"无法从计划文件名识别数据表: {path.name}")
    return match.group(1)


def apply_import_plan(
    client: Any,
    plan_files: list[Path],
    confirm: bool,
) -> ImportSummary:
    if not confirm:
        raise ImportConfirmationRequired("写入木木知识库需要显式确认")
    if not plan_files:
        raise ImportValidationError("没有可执行的导入计划文件")
    manifest_path = plan_files[0].parent.parent / "aitable-manifest.json"
    manifest = load_manifest(manifest_path)
    base_id = str(manifest["base_id"])
    tables = manifest["tables"]
    completed = 0
    record_count = 0
    for path in plan_files:
        table_name = _table_name_from_path(path)
        try:
            table_id = str(tables[table_name]["table_id"])
        except (KeyError, TypeError) as error:
            raise ImportValidationError(f"运行清单缺少数据表: {table_name}") from error
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or len(records) > 30:
            raise ImportValidationError(f"计划文件记录数非法: {path.name}")
        client.create_records(base_id, table_id, path)
        completed += 1
        record_count += len(records)
    return ImportSummary(completed, record_count)


def expected_ids_from_pack(pack_dir: Path) -> dict[str, set[str]]:
    source = json.loads((pack_dir / "source.json").read_text(encoding="utf-8"))
    entries = json.loads((pack_dir / "entries.json").read_text(encoding="utf-8"))
    cases = json.loads((pack_dir / "test-cases.json").read_text(encoding="utf-8"))
    return {
        "知识源台账": {source["source_id"]},
        "知识条目": {entry["knowledge_id"] for entry in entries},
        "木木测试题": {case["test_id"] for case in cases},
    }


def _remote_records(payload: object) -> list[dict[str, object]]:
    value = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def verify_remote_records(
    client: Any,
    manifest: dict[str, object],
    expected_ids: dict[str, set[str]],
) -> VerificationSummary:
    base_id = str(manifest["base_id"])
    table_definitions = manifest["tables"]
    id_fields = {
        "知识源台账": "来源编号",
        "知识条目": "知识编号",
        "木木测试题": "测试编号",
    }
    counts: dict[str, int] = {}
    empty_rows: dict[str, int] = {}
    missing_result: dict[str, tuple[str, ...]] = {}
    duplicate_result: dict[str, tuple[str, ...]] = {}
    for table_name, expected in expected_ids.items():
        definition = table_definitions[table_name]
        table_id = str(definition["table_id"])
        field_id = str(definition["fields"][id_fields[table_name]])
        records = _remote_records(client.query_all_records(base_id, table_id))
        actual: list[str] = []
        for record in records:
            cells = record.get("cells", {})
            if isinstance(cells, dict) and field_id in cells:
                actual.append(str(cells[field_id]))
        counts[table_name] = len(actual)
        empty_rows[table_name] = len(records) - len(actual)
        actual_set = set(actual)
        missing_result[table_name] = tuple(sorted(expected - actual_set))
        duplicate_result[table_name] = tuple(
            sorted(identifier for identifier in actual_set if actual.count(identifier) > 1)
        )
    return VerificationSummary(counts, empty_rows, missing_result, duplicate_result)
